#include "evaluate.hpp"

extern tipl::program_option<tipl::out> po;
using namespace std::chrono_literals;
std::vector<std::string> operations({
        "none",
        "gaussian_filter",
        "smoothing_filter",
        "normalize",
        "upsampling",
        "downsampling",
        "flip_x",
        "flip_y",
        "flip_z",
        "swap_xy",
        "swap_yz",
        "swap_xz"});

size_t linear_cuda(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_param<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound);





template<typename T, typename U>
void subsample8(const T& I, U& subI,size_t sub_index)
{
    tipl::shape<3> sub_dim(I.width()/2,I.height()/2,I.depth()/2);
    subI.resize(sub_dim);
    int skip_x = sub_dim[0]*2;
    int skip_y = sub_dim[1]*2;
    int skip_z = sub_dim[2]*2;
    size_t sub_pos = 0;
    for (tipl::pixel_index<3> index(I.shape()); index < I.size(); ++index)
        if(index[0] != skip_x && index[1] != skip_y && index[2] != skip_z &&
           sub_index == ((index[0] & 1) + ((index[1] & 1) << 1) + ((index[2] & 1) << 2)))
            subI[sub_pos++] = I[index.index()];
}
template<typename T, typename U>
void upsample8(T& I, const U& subI,size_t sub_index)
{
    tipl::shape<3> sub_dim(I.width()/2,I.height()/2,I.depth()/2);
    int skip_x = sub_dim[0]*2;
    int skip_y = sub_dim[1]*2;
    int skip_z = sub_dim[2]*2;
    size_t sub_pos = 0;
    for (tipl::pixel_index<3> index(I.shape()); index < I.size(); ++index)
        if(index[0] != skip_x && index[1] != skip_y && index[2] != skip_z &&
           sub_index == ((index[0] & 1) + ((index[1] & 1) << 1) + ((index[2] & 1) << 2)))
            I[index.index()] = subI[sub_pos++];
}

template<typename out_type,typename template_type,typename image_type>
void reclassify_labels_by_template(const template_type& template_I,image_type& atlas_I)
{
    size_t template_region_count = tipl::max_value(template_I) + 1;
    size_t atlas_region_count = tipl::max_value(atlas_I);
    std::vector<size_t> tissue_votes((atlas_region_count+1)*template_region_count,0);

    size_t sz = atlas_I.size();
    for(size_t pos = 0; pos < sz; ++pos)
    {
        auto a = atlas_I[pos];
        auto t = template_I[pos];
        if(a > 0 && t < template_region_count)
            tissue_votes[a*template_region_count + t]++;
    }

    std::vector<size_t> region_majority_tissue(atlas_region_count+1,0);
    tipl::par_for(atlas_region_count,[&](size_t i)
                  {
                      ++i;
                      auto begin_it = tissue_votes.begin() + i*template_region_count;
                      auto best_tissue = std::max_element(begin_it,begin_it + template_region_count);
                      region_majority_tissue[i] = std::distance(begin_it,best_tissue);
                  });

    std::vector<size_t> region_erased(atlas_region_count+1,0);
    for(size_t pos = 0; pos < sz; ++pos)
    {
        auto a = atlas_I[pos];
        if(a > 0 && template_I[pos] != region_majority_tissue[a])
        {
            atlas_I[pos] = 0;
            region_erased[a]++;
        }
    }

    if constexpr(!std::is_same_v<out_type,void>)
    {
        std::string erased_report;
        for(size_t i = 1; i <= atlas_region_count; ++i)
            if(region_majority_tissue[i] > 0)
            {
                if(!erased_report.empty())
                    erased_report += ", ";
                erased_report += std::to_string(region_erased[i]);
            }

        if(!erased_report.empty())
            out_type() << " voxel erased based on tissue classification: " << erased_report;
    }
}

bool evaluate_unet::load_atlas(const std::string& file_name)
{
    std::string corrected_file_name;
    {
        std::filesystem::path corrected_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toUtf8().constData())/"corrected_atlas";
        if(!std::filesystem::exists(corrected_dir))
            std::filesystem::create_directories(corrected_dir);
        corrected_file_name = (corrected_dir/(tipl::remove_all_suffix(std::filesystem::path(file_name).filename().string()) + ".corrected.nii.gz")).string();
    }

    atlas_I.resize(template_I.shape());
    if(std::filesystem::exists(corrected_file_name))
        return tipl::io::gz_nifti(corrected_file_name,std::ios::in).to_space<tipl::majority>(atlas_I,template_R) >> [&](const std::string& e){error_msg = e;};

    if(!(tipl::io::gz_nifti(file_name,std::ios::in).to_space<tipl::majority>(atlas_I,template_R) >> [&](const std::string& e){error_msg = e;}))
        return false;

    // in template_I is the tissue segmentation, 0 is background, 1: white matter 2: gray matter 3: cerebellar gray matter 4: subcortical
    const size_t template_region_count = 5;
    atlas_region_count = tipl::max_value(atlas_I);

    // zero atlas_I where template_I is zero
    tipl::preserve(atlas_I.begin(),atlas_I.end(),template_I.begin());

    std::vector<size_t> tissue_total;
    tipl::histogram(template_I,tissue_total,0,template_region_count,template_region_count);



    std::vector<float> tissue_coverage(template_region_count,0.0f);
    {
        tipl::progress prog("checking tissue coverage of the atlas");
        std::vector<size_t> atlas_covered(template_region_count,0);
        for(size_t pos = 0,sz = atlas_I.size();pos < sz;++pos)
            if(atlas_I[pos] > 0 && template_I[pos] < template_region_count)
                ++atlas_covered[template_I[pos]];
        for(size_t cur_tissue = 1;cur_tissue < template_region_count;++cur_tissue)
        {
            if(tissue_total[cur_tissue] == 0)
                continue;
            tissue_coverage[cur_tissue] = float(atlas_covered[cur_tissue])/float(tissue_total[cur_tissue]);
            tipl::out() << tissue_names[cur_tissue] << " coverage: " << int(tissue_coverage[cur_tissue]*100.0f) << "%";
        }
    }

    {
        tipl::progress prog("checking tissue classification of each atlas region");
        reclassify_labels_by_template<tipl::out>(template_I,atlas_I);
    }

    {
        tipl::progress prog("region growing to make up missing voxels");
        tipl::image<3,unsigned char> mask(template_I.shape());

        for(size_t cur_tissue = 1;cur_tissue < template_region_count;++cur_tissue)
        {
            if(tissue_coverage[cur_tissue] <= 0.75f)
                continue;
            tipl::out() << "work on " << tissue_names[cur_tissue];
            for(size_t pos = 0,sz = template_I.size();pos < sz;++pos)
                mask[pos] = (template_I[pos] == cur_tissue);
            tipl::morphology::fill_and_smooth_labels<tipl::out>(mask,atlas_I);
        }
    }

    tipl::io::gz_nifti(corrected_file_name,std::ios::out) << template_vs << template_R << true << atlas_I;
    return true;
}

void evaluate_unet::read_file(void)
{
    image_file_name = param.image_file_name;
    device = param.device;
    eval.clear();
    eval.resize(image_file_name.size());
    data_ready = std::vector<unsigned char>(image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        for(size_t i = 0;i < eval.size() && !aborted;++i)
        {
            while(i > cur_prog+6)
                if(aborted) return; else std::this_thread::sleep_for(100ms);
            status = "evaluating";
            eval[i].model_dim = model->dim;
            eval[i].model_vs = model->voxel_size;
            eval[i].in_count = eval[i].cur_count = model->in_count;
            eval[i].out_count = model->out_count;
            eval[i].single_component_label = model->single_component_label;
            eval[i].mask.clear();
            if(!eval[i].load_from_file<tipl::io::gz_nifti>(image_file_name[i]) ||
               !eval[i].run_preproc(model->preproc) ||
               !eval[i].handle_fov_pre(model->fov_strategy) ||
                !eval[i].handle_orientation(model->orientation))
                return tipl::error() << (error_msg = image_file_name[i].u8string() + " : " + eval[i].error_msg),aborted = true,void();
            data_ready[i] = true;
        }
    }));
}

void evaluate_unet::evaluate(void)
{
    cur_prog = 0;
    evaluate_thread.reset(new std::thread([=](){
        try{
            while(cur_prog < eval.size() && !aborted)
            {
                while(!data_ready[cur_prog])
                    if(aborted) return; else std::this_thread::sleep_for(100ms);
                status = "preproc_actions";
                tipl::out() << "inferencing using u-net";
                torch::NoGradGuard no_grad;
                for(size_t i = 0;i < eval[cur_prog].model_io.size();++i)
                {
                    auto& io = eval[cur_prog].model_io[i];
                    auto result = model->forward(torch::from_blob(io.data(),
                                              {1,model->in_count,int(io.depth()/model->in_count),int(io.height()),int(io.width())}).to(device))[0];
                    io.resize(io.shape().multiply(tipl::shape<3>::z,model->out_count).divide(tipl::shape<3>::z,model->in_count));
                    std::memcpy(io.data(),result.to(torch::kCPU).contiguous().data_ptr<float>(),io.size()*sizeof(float));
                }
                cur_prog++;
            }
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error during evaluation:") + error.what();
            aborted = true;
        }
        catch(...)
        {
            aborted = true;
        }
        tipl::out() << error_msg << std::endl;

    }));
}

extern std::vector<std::string> seg_template_list;
extern std::vector<std::vector<std::string> > atlas_file_name_list;;
void evaluate_unet::output(void)
{
    cur_output = 0;
    auto run_evaluation = [this]()
    {
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false;}
        } guard(running);


        try{
            while(cur_output < eval.size() && !aborted)
            {
                while(cur_output >= cur_prog)
                    if(aborted) return; else std::this_thread::sleep_for(100ms);

                status = "evaluating";

                if(eval[cur_output].model_io.empty())
                    return tipl::error() << "no model output for " << image_file_name[cur_output],aborted = true,void();

                if(!eval[cur_output].handle_orientation(model->orientation,true) || !eval[cur_output].handle_fov_post() || !eval[cur_output].run_postproc(model->postproc))
                    return tipl::error() << (error_msg = eval[cur_output].error_msg),aborted = true,void();
                cur_output++;
            }
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error during output:") + error.what();
        }
        catch(...)
        {
        }
        if(!error_msg.empty())
            tipl::error() << error_msg;
        aborted = true;
        status = "complete";
    };

    if(tipl::show_prog)
        output_thread.reset(new std::thread(run_evaluation));
    else
    {
        run_evaluation();
        join();
    }
}



template<typename image_type>
void postproc_actions(const std::string& command,
                      float param1,float param2,
                      image_type& this_image,
                      image_type& mask,
                      const tipl::shape<3>& image_dim,
                      char& is_label)
{
    auto out_channel = this_image.depth()/image_dim[2];
    if(this_image.empty())
        return;
    tipl::out() << "run " << command;
    if(command == "argmax")
    {
        this_image = tipl::argmax(this_image,mask.shape(),(mask > param1).data());
        is_label = true;
    }

    // per channel operations
    tipl::par_for(out_channel,[&](size_t label)
    {
        auto I = this_image.alias(image_dim.size()*label,image_dim);
        if(command == "upper_threshold")
        {
            float upper_threshold_threshold = param1;
            tipl::upper_threshold(I,upper_threshold_threshold);
            is_label = false;
            return;
        }
        if(command == "lower_threshold")
        {
            float lower_threshold_threshold = param1;
            tipl::lower_threshold(I,lower_threshold_threshold);
            is_label = false;
            return;
        }
        if(command == "minus")
        {
            float minus_value = param1;
            for(size_t i = 0,sz = I.size();i < sz;++i)
                I[i] -= minus_value;
            is_label = false;
            return;
        }

        if(command == "defragment_each")
        {
            float defragment_each_threshold = param1;
            tipl::image<3,char> mask(I.shape()),mask2;
            for(size_t i = 0,sz = I.size();i < sz;++i)
                mask[i] = (I[i] > defragment_each_threshold ? 1:0);
            mask2 = mask;
            tipl::morphology::defragment_by_size_ratio(mask);
            for(size_t i = 0,sz = I.size();i < sz;++i)
                if(!mask[i] && mask2[i])
                I[i] = 0;
            return;
        }
        if(command == "normalize_each")
        {
            tipl::normalize(I);
            is_label = false;
            return;
        }
        if(command == "gaussian_smoothing")
        {
            tipl::filter::gaussian(I);
            is_label = false;
            return;
        }
    });

    tipl::error() << "unknown command " << command << std::endl;
}


void evaluate_unet::proc_actions(const char* cmd,float param1,float param2)
{
    char is_label;
    postproc_actions(cmd,param1,param2,eval[cur_output].label_prob,eval[cur_output].fg_prob,eval[cur_output].image_dim,is_label);
}


void evaluate_unet::start(void)
{
    status = "initiating";
    stop();
    model->to(param.device);
    model->eval();    
    for(auto& m : model->modules())
        if(auto bn = std::dynamic_pointer_cast<torch::nn::BatchNorm3dImpl>(m))
        {
            bn->eval();
            bn->running_mean.zero_();
            bn->running_var.fill_(1.0f);
            if(bn->num_batches_tracked.defined())
                bn->num_batches_tracked.zero_();
        }
    aborted = false;
    running = true;
    error_msg.clear();
    read_file();
    evaluate();
    output();
}

void evaluate_unet::join(void)
{
    if(read_file_thread.get())
    {
        read_file_thread->join();
        read_file_thread.reset();
    }
    if(evaluate_thread.get())
    {
        evaluate_thread->join();
        evaluate_thread.reset();
    }
    if(output_thread.get())
    {
        output_thread->join();
        output_thread.reset();
    }
}
void evaluate_unet::stop(void)
{
    aborted = true;
    join();
}
bool evaluate_unet::save_to_file(size_t currentRow,const std::filesystem::path& file_name,unsigned char output)
{
    if(currentRow >= eval.size())
        return false;
    tipl::io::gz_nifti out;
    if(!out.open(file_name,std::ios::out))
        return error_msg = out.error_msg,false;
    out << eval[currentRow].untouched_srow << eval[currentRow].image_vs;
    auto save = [&](auto data)->bool
    {
        tipl::io::apply_flip_swap_seq(data,eval[currentRow].flip_swap,true);
        if(data.depth() == eval[currentRow].image_dim[2])
            return out << data;
        else
            return out << data.alias(0,tipl::shape<4>(eval[currentRow].image_dim.expand(data.depth()/eval[currentRow].image_dim[2])));
    };
    switch(output)
    {
    case 0: // 3d label
        return save(eval[currentRow].label);
    case 1: // skull strip
        {
            tipl::image<3> I;
            if(!(tipl::io::gz_nifti(image_file_name[currentRow],std::ios::in)
                  >> I >> [&](const std::string& e){error_msg = e;}))
                return false;
            return save(I *= eval[currentRow].fg_prob);
        }
    case 2: // mask
        return save(eval[currentRow].fg_prob);
    case 3: // 4d prob
        return save(eval[currentRow].label_prob);
    }
    return false;
}

bool load_from_file(UNet3d& model,const char* file_name);
std::string get_model_path(void);
int eval(void)
{
    static evaluate_unet eval;
    if(eval.running)
    {
        tipl::out() << "terminating training...";
        eval.stop();
    }


    // loading images data
    if(!po.check("source"))
        return 1;
    if((eval.param.image_file_name = po.get_files("source")).empty())
        return tipl::error() << "no file specified at --source",1;

    auto model_path = get_model_path();
    {
        if(!std::filesystem::exists(model_path))
            return tipl::error() << "cannot find the network file " << model_path,1;
        tipl::out() << "loading network " << model_path;
        if(!load_from_file(eval.model,model_path.c_str()))
            return tipl::error() << "failed to load model from " << model_path,1;
        tipl::out() << eval.model->get_info();
    }

    if(po.has("template") && po.has("atlas"))
    {
        size_t seg_id = po.get("template",seg_template_list,0);
        if(seg_id >= seg_template_list.size())
            return tipl::error() << "invalid template",1;
        if(!eval.load_template(seg_template_list[seg_id]) ||
           !eval.load_atlas(po.get("atlas",atlas_file_name_list[seg_id],atlas_file_name_list[seg_id][0])))
            return tipl::error() << eval.error_msg,1;
    }
    eval.param.device = torch::Device(po.get("device",torch::hasCUDA() ? "cuda:0" :
                                                       (torch::hasHIP() ? "hip:0" :
                                                       (torch::hasMPS() ? "mps:0": "cpu"))));
    {
        tipl::progress p("start evaluating");
        eval.start();
        eval.join();
    }

    if(!eval.error_msg.empty())
        return tipl::error() << eval.error_msg,1;
    tipl::progress p("saving results");
    for(size_t i = 0;p(i,eval.param.image_file_name.size());++i)
    {
        auto file_name = std::filesystem::path(eval.param.image_file_name[i])+=".result.nii.gz";
        tipl::out() << "save to " << file_name;
        if(!eval.save_to_file(i,file_name,po.get("output_type",0)))
            return tipl::error() << eval.error_msg,1;
    }
    return 0;
}
