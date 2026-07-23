#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include "train.hpp"

extern tipl::program_option<tipl::out> po;

namespace
{

struct qc_stat
{
    uint64_t voxels = 0,wrong = 0;

    qc_stat& operator+=(const qc_stat& rhs)
    {
        voxels += rhs.voxels;
        wrong += rhs.wrong;
        return *this;
    }
    double ratio(void) const
    {
        return voxels ? double(wrong)/voxels : 0.0;
    }
};

struct qc_case
{
    tipl::image<3> image,label;
};

void write_case(std::ostream& out,
                const std::string& image,
                const std::string& label,
                const std::vector<qc_stat>& stats,
                const qc_stat& overall,
                size_t unavailable_before = 0)
{
    out << std::filesystem::path(image).filename().string() << '\t'
        << std::filesystem::path(label).filename().string() << '\t'
        << overall.ratio();

    for(size_t c = 0;c < stats.size();++c)
        if(c < unavailable_before)
            out << "\tN/A";
        else
            out << '\t' << stats[c].ratio();

    out << '\n';
}

bool calculate_qc(UNet3d& model,
                  const torch::Device& device,
                  qc_case& data,
                  size_t collapse_before,
                  std::vector<qc_stat>& stats,
                  qc_stat& overall,
                  std::string& error_msg)
{
    const int64_t raw_C = model->out_count;
    const int64_t D = model->dim[2];
    const int64_t H = model->dim[1];
    const int64_t W = model->dim[0];
    int64_t C = raw_C;

    if(data.image.size() != model->dim.size()*size_t(model->in_count) ||
        data.label.size() != model->dim.size())
        return error_msg = "training data dimension mismatch",false;

    if(collapse_before >= size_t(raw_C))
        return error_msg = "invalid collapse_before",false;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    auto input = torch::from_blob(
                     data.image.data(),
                     {1,model->in_count,D,H,W},
                     options).to(device);

    auto target = torch::from_blob(
                      data.label.data(),
                      {1,D,H,W},
                      options).to(device).to(torch::kLong);

    auto outputs = model->forward(input);
    if(outputs.empty() || !outputs[0].defined())
        return error_msg = "undefined model output",false;

    auto logits = std::move(outputs[0]);
    outputs.clear();
    input = torch::Tensor();

    if(logits.dim() != 5 ||
        logits.size(1) != raw_C ||
        logits.size(2) != D ||
        logits.size(3) != H ||
        logits.size(4) != W)
        return error_msg = "model output dimension mismatch",false;

    auto valid = target.ge(0).logical_and(target.lt(raw_C));

    if(collapse_before)
    {
        logits = torch::cat({
                                torch::logsumexp(
                                    logits.slice(1,0,int64_t(collapse_before)),1,true),
                                logits.slice(1,int64_t(collapse_before),raw_C)
                            },1);

        target = torch::clamp_min(
            target-int64_t(collapse_before)+1,0);

        C -= int64_t(collapse_before)-1;
    }

    auto safe_target = target.clamp(0,C-1);
    auto label_bin = torch::where(
                         valid,
                         safe_target,
                         torch::full_like(safe_target,C)).reshape({-1});

    auto wrong = logits.argmax(1)
                     .ne(target)
                     .logical_and(valid)
                     .reshape({-1})
                     .to(torch::kFloat32);

    logits = torch::Tensor();

    auto packed = torch::stack({
                                   label_bin.bincount({},C+1).to(torch::kFloat64),      // voxel counts
                                   label_bin.bincount(wrong,C+1).to(torch::kFloat64)    // wrong counts
                               }).to(torch::kCPU).contiguous();

    const auto* value = packed.data_ptr<double>();
    const size_t stride = size_t(C+1);

    stats.assign(size_t(raw_C),{});
    overall = {};

    for(size_t c = 0;c < size_t(C);++c)
    {
        qc_stat stat{
            uint64_t(value[c]),
            uint64_t(value[stride+c])
        };

        overall += stat;

        if(!collapse_before)
            stats[c] = stat;
        else if(c)
            stats[collapse_before+c-1] = stat;
    }

    return true;
}

} // namespace

int qc(void)
{
    namespace fs = std::filesystem;

    if(!po.has("bids"))
        return tipl::error() << "please specify --bids",1;

    std::vector<std::string> images,labels;
    if(!get_bids_pairs(po.get("bids"),images,labels))
        return 1;

    auto model_path = get_model_path();
    if(!fs::exists(model_path))
        return tipl::error() << "cannot find model " << model_path,1;

    UNet3d model;
    if(!load_from_file(model,model_path.c_str()))
        return tipl::error() << "cannot load model " << model_path,1;

    if(model->out_count < 2)
        return tipl::error() << "QC requires a categorical model",1;

    torch::Device device(po.get(
        "device",
        torch::hasCUDA() ? "cuda:0" :
            torch::hasHIP()  ? "hip:0" :
            torch::hasMPS()  ? "mps:0" : "cpu"));

    model->prepare_for_inference(device);

    std::unordered_map<std::string,std::pair<bool,int>> label_info;
    size_t max_template_label = 0;
    std::string error_msg;

    {
        tipl::progress prog("reading label information");

        for(size_t i = 0;prog(i,labels.size());++i)
        {
            auto [iter,inserted] = label_info.try_emplace(labels[i]);

            if(inserted &&
                !read_label_info(
                    labels[i],
                    iter->second.first,
                    iter->second.second,
                    error_msg))
                return tipl::error() << error_msg,1;

            if(iter->second.first)
                max_template_label = std::max(
                    max_template_label,
                    size_t(iter->second.second));
        }

        if(tipl::progress::aborted())
            return 1;
    }

    if(!max_template_label)
    {
        tipl::warning() << "no template label found; use default 5";
        max_template_label = 5;
    }

    std::vector<char> shift_label(labels.size());

    for(size_t i = 0;i < labels.size();++i)
    {
        const auto& [is_template,max_label] =
            label_info.at(labels[i]);

        shift_label[i] =
            !is_template &&
            max_label < int(max_template_label) &&
            max_label+int(max_template_label) < model->out_count;
    }

    auto load_case = [&](size_t i,qc_case& data)
    {
        if(!read_image_and_label(
                images[i],labels[i],
                model->dim,model->voxel_size,
                data.image,data.label))
            return false;

        if(shift_label[i])
            shift_subject_label(
                data.image,data.label,max_template_label);

        return true;
    };

    std::vector<std::vector<qc_stat>> case_stats(images.size());
    std::vector<qc_stat> case_overall(images.size());
    std::vector<std::string> case_error(images.size());

    const size_t worker_count = std::min({
        size_t(4),
        size_t(std::max(1,po.get("thread_count",4))),
        images.size()
    });

    std::atomic_size_t next_case{0},progress{0};
    std::atomic_bool failed{false};

    {
        tipl::progress prog("evaluating training data");

        tipl::par_for(worker_count,[&](size_t)
            {
                torch::NoGradGuard no_grad;
                torch::DeviceGuard device_guard(device);

                while(!failed && !tipl::progress::aborted())
                {
                    const size_t i = next_case++;
                    if(i >= images.size())
                        break;

                    try
                    {
                        qc_case data;

                        if(load_case(i,data) &&
                            calculate_qc(
                                model,
                                device,
                                data,
                                shift_label[i] ? max_template_label+1 : 0,
                                case_stats[i],
                                case_overall[i],
                                case_error[i]))
                        {
                            prog(++progress,images.size());
                            continue;
                        }

                        if(case_error[i].empty())
                            case_error[i] = "cannot read image or label";
                    }
                    catch(const std::exception& e)
                    {
                        case_error[i] = e.what();
                    }
                    catch(...)
                    {
                        case_error[i] = "unknown QC error";
                    }

                    failed = true;
                    break;
                }
            },worker_count);

        if(tipl::progress::aborted())
            return 1;
    }

    if(failed)
    {
        for(size_t i = 0;i < case_error.size();++i)
            if(!case_error[i].empty())
                return tipl::error()
                           << images[i] << ": " << case_error[i],1;

        return tipl::error() << "QC failed",1;
    }

    fs::path model_file(model_path);
    fs::path report =
        model_file.parent_path()/
        (model_file.stem().string()+".error_report.tsv");
    fs::path temporary(report.string()+".tmp");

    std::ofstream out(temporary);
    if(!out)
        return tipl::error() << "cannot write " << temporary,1;

    out << std::setprecision(9)
        << "image\tground_truth\twrong_ratio";

    for(int c = 0;c < model->out_count;++c)
        out << "\twrong_ratio" << c;

    out << '\n';

    for(size_t i = 0;i < images.size();++i)
        write_case(
            out,
            images[i],
            labels[i],
            case_stats[i],
            case_overall[i],
            shift_label[i] ? max_template_label+1 : 0);

    out.close();
    if(!out)
        return tipl::error()
                   << "failed writing " << temporary,1;

    std::error_code ec;
    fs::remove(report,ec);

    if(!ec)
        fs::rename(temporary,report,ec);

    if(ec)
        return tipl::error()
                   << "cannot create " << report
                   << ": " << ec.message(),1;

    tipl::out() << "QC report saved to " << report;
    return 0;
}
