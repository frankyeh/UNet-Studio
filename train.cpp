#include <random>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <QFile>
#include <QTextStream>
#include "train.hpp"

extern tipl::program_option<tipl::out> po;
using namespace std::chrono_literals;

bool load_from_file(UNet3d& model,const char* file_name);
bool save_to_file(UNet3d& model,const char* file_name);

bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,tipl::image<3>& input,tipl::image<3>& label)
{
    std::scoped_lock<std::mutex> lock(tipl::io::nifti_do_not_show_process);
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    tipl::shape<3> image_dim;
    if(!(tipl::io::gz_nifti(image_name,std::ios::in) >> image_dim >> image_t >> input >> [&](const std::string& e)
          {tipl::error() << e;}))
        return false;
    label.clear();
    label.resize(image_dim);
    return tipl::io::gz_nifti(label_name,std::ios::in).to_space<tipl::majority>(label,image_t) >>
           [&](const std::string& e){tipl::error() << e;};
}



void preprocessing(tipl::image<3>& image,tipl::image<3>& label,tipl::shape<3> to_dim)
{
    tipl::shape<3> from_dim(label.shape());
    if(from_dim!=to_dim)
    {
        auto shift = tipl::vector<3,int>(to_dim)-tipl::vector<3,int>(from_dim);
        shift /= 2;
        tipl::image<3> new_label(to_dim);
        tipl::draw(label,new_label,shift);
        new_label.swap(label);

        int in_count = image.depth()/from_dim[2];
        tipl::image<3> new_image(to_dim.multiply(tipl::shape<3>::z,in_count));
        size_t from_sz = from_dim.size(),to_sz = to_dim.size();

        for(int c = 0;c<in_count;++c)
        {
            auto from = image.alias(c*from_sz,from_dim);
            auto to = new_image.alias(c*to_sz,to_dim);
            tipl::draw(from,to,shift);
            tipl::normalize(to);
        }
        new_image.swap(image);
    }
}

void simulate_modality(tipl::image<3>& t1w,
                       const tipl::image<3>& label,
                       unsigned int max_label,
                       unsigned int seed)
{
    // t1w is already normalized to [0,1].
    // label stores integer values 0..max_label.
    constexpr size_t term_count = 20;

    tipl::uniform_dist<int> rand_int(seed);
    tipl::uniform_dist<float> rand_float(0.0f,1.0f,seed+1);

    tipl::image<3> tissue(label.shape());
    std::vector<float> lut(max_label+1);
    for(auto& v : lut)
        v = 0.4f + rand_float()*0.2f;
    for(size_t i = 0;i < label.size();++i)
        tissue[i] = lut[int(label[i])];

    tipl::filter::gaussian(tissue);
    tipl::filter::gaussian(tissue);

    struct term_type { uint8_t a,b,c,d; float w; };
    std::array<term_type,term_count> terms;
    for(auto& t : terms)
    {
        do
        {
            t.a = uint8_t(rand_int(4));
            t.b = uint8_t(rand_int(4));
        }
        while(t.a+t.b == 0);
        t.c = uint8_t(rand_int(4));
        t.d = uint8_t(rand_int(4));
        t.w = rand_float();
    }

    const float gamma = 0.6f + 1.2f*rand_float();
    float mn = std::numeric_limits<float>::max();
    float mx = -mn;

    for(size_t i = 0;i < t1w.size();++i)
    {
        const float x = t1w[i];
        if(x <= 0.02f)
        {
            t1w[i] = 0.0f;
            continue;
        }

        const float z = tissue[i], rx = 1.0f-x, rz = 1.0f-z;
        const float px[4] = {1.0f,x,x*x,x*x*x};
        const float pz[4] = {1.0f,z,z*z,z*z*z};
        const float qx[4] = {1.0f,rx,rx*rx,rx*rx*rx};
        const float qz[4] = {1.0f,rz,rz*rz,rz*rz*rz};

        float s = 0.0f;
        for(const auto& t : terms)
            s += t.w*px[t.a]*pz[t.b]*qx[t.c]*qz[t.d];

        t1w[i] = std::pow(s,gamma);
        if(label[i])
        {
            mn = std::min(mn,t1w[i]);
            mx = std::max(mx,t1w[i]);
        }
    }

    if(mx > mn)
    {
        t1w -= mn;
        t1w *= 1.0f/(mx-mn);
        tipl::upper_lower_threshold(t1w,0.0f,1.0f);
    }
}

void train_unet::read_file(void)
{
    read_images.reset(new std::thread([this]()
    {
        std::vector<char> train_image_is_template(std::vector<char>(param.image_file_name.size(),false));

        for(size_t i = 0,sz = param.image_file_name.size();i<sz;++i)
        {
            reading_status = "checking "+ param.image_file_name[i];
            bool is_mni = false;
            if(!(tipl::io::gz_nifti(param.image_file_name[i].c_str(),std::ios::in) >> is_mni >>
                [&](const std::string& e){error_msg = e,aborted = true;}))
                return;
            if((train_image_is_template[i] = is_mni))
            {
                tipl::out() << "template found: " << param.image_file_name[i];
                param.test_image_file_name.push_back(param.image_file_name[i]);
                param.test_label_file_name.push_back(param.label_file_name[i]);
            }
        }

        std::vector<size_t> template_indices;
        std::vector<size_t> non_template_indices;
        for(size_t i = 0,sz = train_image_is_template.size();i<sz;++i)
        {
            if(train_image_is_template[i])
                template_indices.push_back(i);
            else
                non_template_indices.push_back(i);
        }

        tipl::out() << "a total of " << param.image_file_name.size() << " training dataset\n";
        tipl::out() << "a total of " << param.test_image_file_name.size() << " testing dataset\n";

        for(int read_id = 0,sz = param.test_image_file_name.size();read_id<sz && !aborted;++read_id)
        {
            while(pause)
                if(aborted) return; else std::this_thread::sleep_for(100ms);

            reading_status = "reading "+std::filesystem::path(param.test_image_file_name[read_id]).filename().string();

            tipl::image<3> input_image,input_label;
            if(!read_image_and_label(param.test_image_file_name[read_id],
                                     param.test_label_file_name[read_id],input_image,input_label))
                return error_msg = "cannot read image or label data for "+std::filesystem::path(param.test_image_file_name[read_id]).filename().string(),aborted = true,void();

            preprocessing(input_image,input_label,model->dim);

            if(model->out_count==1)
                tipl::normalize(input_label);

            try
            {
                test_in_tensor.push_back(torch::from_blob(input_image.data(),{1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).clone().to(param.test_device,true));
                test_out_tensor.push_back(torch::from_blob(input_label.data(),{1,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(torch::kLong).clone().to(param.test_device,true));
            }
            catch(const c10::Error& error)
            {
                error_msg = std::string("test tensor allocation error: ")+error.what();
                aborted = true;
                return;
            }
        }
        test_data_ready = true;

        std::uniform_int_distribution<int> template_gen(0,std::max<int>(1,template_indices.size())-1);
        std::uniform_int_distribution<int> non_template_gen(0,std::max<int>(1,non_template_indices.size())-1);
        std::mt19937 gen(param.seed);
        size_t begin_epoch = param.batch_size*(model->errors.size()/3);
        for(size_t seed_id = 0;!aborted;++seed_id)
        {
            bool use_template = non_template_indices.empty() || seed_id % param.batch_size < template_indices.size();
            size_t read_id = (use_template ? template_indices[template_gen(gen)] : non_template_indices[non_template_gen(gen)]);
            if(seed_id < begin_epoch)
                continue;
            size_t thread = seed_id % in_file.size();
            tipl::image<3> image,label;

            if(train_image[read_id].empty())
            {
                reading_status = "reading "+std::filesystem::path(param.image_file_name[read_id]).filename().string()+
                                 " and "+std::filesystem::path(param.label_file_name[read_id]).filename().string();
                if(!read_image_and_label(param.image_file_name[read_id],param.label_file_name[read_id],image,label))
                    return error_msg = "cannot read image or label data for "+std::filesystem::path(param.image_file_name[read_id]).filename().string(),aborted = true,void();
                reading_status = "preprocessing";
                preprocessing(image,label,model->dim);
                if(!param.is_label)
                    tipl::normalize(label);
                if(train_image_is_template[read_id])
                {
                    train_image[read_id] = image;
                    train_label[read_id] = label;
                }
            }
            else
            {
                reading_status = "using template";
                image = train_image[read_id];
                label = train_label[read_id];
            }

            while(file_ready[thread])
                if(aborted) return; else std::this_thread::sleep_for(100ms);

            in_file[thread].swap(image);
            out_file[thread].swap(label);
            in_file_seed[thread] = seed_id;
            in_file_read_id[thread] = read_id;
            file_ready[thread] = true;
        }
        reading_status = "reading completed";
    }));

    augmentation_thread.reset(new std::thread([=]()
    {
        std::mutex m;
        tipl::par_for(in_data.size(),[&](size_t thread)
        {
            while(!aborted)
            {
                while(!file_ready[thread]||pause)
                    if(aborted) return; else std::this_thread::sleep_for(100ms);

                tipl::image<3> in_data_thread,out_data_thread;
                size_t read_id = in_file_read_id[thread];

                simulate_modality(in_file[thread],out_file[thread],model->out_count,in_file_seed[thread]);

                in_data_thread.swap(in_file[thread]);
                out_data_thread.swap(out_file[thread]);
                file_ready[thread] = false;

                {
                    std::lock_guard<std::mutex> lock(m);
                    augmentation_status = "augmenting "+std::filesystem::path(param.image_file_name[read_id]).filename().string();
                }

                visual_perception_augmentation(param.options,in_data_thread,out_data_thread,param.is_label,model->dim,in_file_seed[thread]);

                while(data_ready[thread]||pause)
                    if(aborted) return; else std::this_thread::sleep_for(100ms);

                in_data[thread].swap(in_data_thread);
                out_data[thread].swap(out_data_thread);
                in_data_read_id[thread] = read_id;
                data_ready[thread] = true;
            }
        },thread_count);
        augmentation_status = "augmentation completed";
    }));
}

std::string train_unet::get_status(void)
{
    std::string s1,s2;
    s1.resize(file_ready.size());
    s2.resize(data_ready.size());
    for(size_t i = 0,sz = file_ready.size();i<sz;++i)
    {
        s1[i] = file_ready[i]?'-':'_';
        s2[i] = data_ready[i]?'-':'_';
    }
    return s1+"|"+s2;
}

inline std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calc_losses(const torch::Tensor& pred_raw,const torch::Tensor& target_indices,int C)
{
    auto ce = torch::nn::functional::cross_entropy(pred_raw,target_indices);
    auto pred_probs = torch::clamp(torch::softmax(pred_raw,1),1e-6,1.0-1e-6);

    auto target_one_hot = torch::nn::functional::one_hot(target_indices,C).permute({0,4,1,2,3}).to(pred_probs.dtype());
    auto pred_fg = pred_probs.slice(1,1,C);
    auto target_fg = target_one_hot.slice(1,1,C);

    auto inter = torch::sum(pred_fg*target_fg,{2,3,4});
    auto card = torch::sum(pred_fg+target_fg,{2,3,4});

    auto eps = torch::tensor(1e-5,torch::TensorOptions().device(pred_raw.device()).dtype(pred_raw.dtype()));
    auto dice = 1.0f-torch::mean((2.0f*inter+eps)/(card+eps));
    auto mse = torch::mse_loss(pred_probs,target_one_hot)*static_cast<float>(C);

    return {ce,dice,mse};
}

void train_unet::train(void)
{
    auto run_training = [=]()
    {
        try
        {
            if(cur_epoch == 0 && model->prior_errors.empty())
            {
                model->report += " Training was conducted over "+std::to_string(param.epoch)+" epochs ";
                model->report += "using a batch size of "+std::to_string(param.batch_size)+". ";
                model->report += "Optimization employed an initial learning rate of "+std::to_string(param.learning_rate)+" using SGD with Nesterov momentum.";
            }

            while(cur_epoch<param.epoch && !aborted)
            {
                size_t cur_data_index = cur_epoch*param.batch_size;
                training_status = "training";
                double cur_lr = param.learning_rate*std::pow(1.0-(double)cur_epoch/param.epoch,0.9);
                for(auto& group:model->optimizer->param_groups())
                {
                    auto& opt = static_cast<torch::optim::SGDOptions&>(group.options());
                    opt.lr(cur_lr);
                }

                for(auto& each:other_models)
                {
                    each->copy_from(*model);
                    for(auto& p:each->parameters())
                        if(p.grad().defined())
                            p.grad().zero_();
                }

                int total_gpus = 1+other_models.size();
                int active_threads = std::min<int>(total_gpus,param.batch_size);
                std::atomic<int> next_batch_idx{0};

                std::vector<std::thread> gpu_threads;
                std::mutex out_mutex;

                for(int thread_id = 0;thread_id < active_threads;++thread_id)
                {
                    gpu_threads.emplace_back([&,thread_id]()
                    {
                        try
                        {
                            auto cur_model = thread_id == 0 ? model : other_models[thread_id-1];
                            auto dev = cur_model->device();
                            torch::DeviceGuard guard(dev);

                            while(!aborted)
                            {
                                int b = next_batch_idx.fetch_add(1);
                                if(b >= param.batch_size)
                                    break;

                                size_t data_idx = (cur_data_index+b)%data_ready.size();
                                while(!data_ready[data_idx] || pause)
                                    if(aborted) return; else std::this_thread::sleep_for(100ms);


                                auto target_cpu = torch::from_blob(
                                    out_data[data_idx].data(),
                                    {1,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).clone().to(torch::kLong);

                                auto in_cpu = torch::from_blob(
                                    in_data[data_idx].data(),
                                    {1,cur_model->in_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).clone();

                                data_ready[data_idx] = false;

                                auto target = target_cpu.to(dev);
                                auto in = in_cpu.to(dev);

                                auto outputs = cur_model->forward(in);
                                in = torch::Tensor();

                                torch::Tensor active_target = target;
                                target = torch::Tensor();

                                torch::Tensor total_loss;
                                size_t out_sz = outputs.size();

                                float weight_sum = 0.0f;
                                for(size_t k = 0;k < out_sz;++k)
                                    weight_sum += 1.0f/(1 << k);

                                float inv_weight_sum = 1.0f/weight_sum;

                                for(size_t k = 0;k < out_sz;++k)
                                {
                                    if(k > 0)
                                    {
                                        int64_t d = active_target.size(1) >> 1;
                                        int64_t h = active_target.size(2) >> 1;
                                        int64_t w = active_target.size(3) >> 1;

                                        if(d <= 0 || h <= 0 || w <= 0)
                                            throw std::runtime_error("deep supervision target size became zero");

                                        auto temp_float = active_target.unsqueeze(1).to(torch::kFloat32);
                                        auto opt = torch::nn::functional::InterpolateFuncOptions()
                                            .size(std::vector<int64_t>{d,h,w})
                                            .mode(torch::kNearest);

                                        active_target = torch::nn::functional::interpolate(temp_float,opt)
                                            .squeeze(1)
                                            .to(torch::kLong);
                                    }

                                    if(!outputs[k].defined())
                                        throw std::runtime_error("undefined deep supervision output at level " + std::to_string(k));

                                    if(outputs[k].size(1) != cur_model->out_count)
                                        throw std::runtime_error(
                                            "output channel mismatch at level " + std::to_string(k) +
                                            ": tensor has " + std::to_string(outputs[k].size(1)) +
                                            ", out_count is " + std::to_string(cur_model->out_count));

                                    auto max_label = active_target.max().item<int64_t>();
                                    if(max_label >= cur_model->out_count)
                                        throw std::runtime_error(
                                            "target label out of range at level " + std::to_string(k) +
                                            ": max label=" + std::to_string(max_label) +
                                            ", out_count=" + std::to_string(cur_model->out_count));

                                    auto [ce,dice,mse] = calc_losses(outputs[k],active_target,cur_model->out_count);
                                    outputs[k] = torch::Tensor();

                                    float norm_weight = (1.0f/(1 << k))*inv_weight_sum;

                                    torch::Tensor level_loss;
                                    if(param.cost_ce)
                                        level_loss = level_loss.defined() ? level_loss + ce : ce;
                                    if(param.cost_dice)
                                        level_loss = level_loss.defined() ? level_loss + dice : dice;
                                    if(param.cost_mse)
                                        level_loss = level_loss.defined() ? level_loss + mse : mse;

                                    if(!level_loss.defined())
                                        level_loss = ce;

                                    level_loss *= norm_weight;
                                    total_loss = total_loss.defined() ? total_loss + level_loss : level_loss;
                                }

                                if(!total_loss.defined())
                                    throw std::runtime_error("undefined total loss");

                                total_loss.backward();
                            }
                        }
                        catch(const c10::Error& e)
                        {
                            std::scoped_lock<std::mutex> lock(out_mutex);
                            tipl::error() << (error_msg = std::string("GPU thread ") + std::to_string(thread_id) + ": " + e.what());
                            aborted = true;
                        }
                    });
                }

                for(auto& t:gpu_threads)
                    t.join();

                if(aborted)
                    return;

                training_status = "update model";
                for(auto& each:other_models)
                    model->add_gradient_from(*each);

                for(auto& p:model->parameters())
                    if(p.grad().defined())
                        p.grad().div_(param.batch_size);

                torch::nn::utils::clip_grad_norm_(model->parameters(), 12.0);

                model->optimizer->step();
                model->optimizer->zero_grad();

                // wait for validation thread to finish last epoch
                training_status = "waiting for validation";
                while(cur_validation_epoch < cur_epoch)
                    if(aborted) return; else std::this_thread::sleep_for(100ms);

                {
                    std::scoped_lock<std::mutex> lock(output_model_mutex);
                    output_model->copy_from(*model);
                }

                ++cur_epoch;

                if(save_model_during_training && !model_path.empty() && (cur_epoch % 100 == 0))
                {
                    training_status = "saving model";
                    while(cur_validation_epoch < cur_epoch)
                        if(aborted) return;else std::this_thread::sleep_for(100ms);
                    tipl::out() << "saving model to " << model_path;
                    save_to_file(model,model_path.c_str());
                    torch::save(*(model->optimizer),(std::filesystem::path(model_path) += ".opt").make_preferred().string());
                }
            }
        }
        catch(const c10::Error& e)
        {
            error_msg = std::string("during ")+training_status+":"+e.what();
            tipl::out() << error_msg;
        }
        catch(...)
        {
            error_msg = "unknown error in training";
            tipl::out() << error_msg;
        }
        pause = true;
        aborted = true;
    };
    train_thread.reset(new std::thread(run_training));
}

void train_unet::validate(void)
{
    auto run_validation = [=]()
    {
        try
        {
            struct exist_guard
            {
                bool& running;
                exist_guard(bool& running_):running(running_){}
                ~exist_guard()
                {
                    running = false;
                }
            } guard(running);

            auto start_time = std::chrono::steady_clock::now();
            size_t start_validation_epoch = cur_validation_epoch;

            for(;cur_validation_epoch<param.epoch&&!aborted;++cur_validation_epoch)
            {
                while(cur_epoch <= cur_validation_epoch || !test_data_ready)
                    if(aborted) return; else std::this_thread::sleep_for(100ms);

                std::vector<float> errors;
                if(!test_in_tensor.empty())
                {
                    std::scoped_lock<std::mutex> lock(output_model_mutex);
                    torch::NoGradGuard no_grad;
                    output_model->eval();
                    double ce_v(0.0),dice_v(0.0),mse_v(00.0);
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                    {
                        auto [ce,dice,mse] = calc_losses(output_model->forward(test_in_tensor[i])[0],test_out_tensor[i],output_model->out_count);
                        ce_v += ce.item().toFloat();
                        dice_v += dice.item().toFloat();
                        mse_v += mse.item().toFloat();
                    }

                    {
                        errors.push_back(ce_v/double(test_in_tensor.size()));
                        errors.push_back(dice_v/double(test_in_tensor.size()));
                        errors.push_back(mse_v/double(test_in_tensor.size()));
                    }

                }
                {
                    if(!cur_validation_epoch)
                        tipl::out() << "1                                                   0.1                                               0.01";

                    if(cur_validation_epoch%100==0)
                    {
                        std::string out = "|-------------------------|--------------------------|-------------------------|-------------------------|";
                        double cur_lr = param.learning_rate*std::pow(1.0-(double)cur_validation_epoch/param.epoch,0.9);
                        auto str = "lr:"+std::to_string(cur_lr);
                        if(cur_validation_epoch>start_validation_epoch)
                        {
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                               std::chrono::steady_clock::now()-start_time).count();
                            auto done = cur_validation_epoch-start_validation_epoch;
                            auto fmt = [](auto s){return std::to_string(s/3600)+"h"+std::to_string((s%3600)/60)+"m";};
                            str += "," + fmt(elapsed*(param.epoch-cur_validation_epoch)/done) +
                                   "/" + fmt(elapsed*param.epoch/done);
                        }
                        size_t copy_len = std::min(str.length(),out.length()-2);
                        std::copy(str.begin(),str.begin()+copy_len,out.begin()+1);
                        tipl::out() << out;
                    }

                    std::string out = "|                         |                          |                         |                         |";
                    if(!errors.empty())
                    {
                        auto to_chart = [&](float error)->int
                        {
                            return int(std::max<float>(0.0f,std::min<float>(float(out.size()-1),(-std::log10(error))*float(out.size()-1)/2.0f)));
                        };
                        out[to_chart(errors[0])] = 'C';
                        out[to_chart(errors[1])] = 'D';
                        out[to_chart(errors[2])] = 'M';
                    }
                    tipl::out() << out << cur_validation_epoch;

                    std::scoped_lock<std::mutex> lock(error_mutex);
                    for(auto each : errors)
                    {
                        model->errors.push_back(each);
                        output_model->errors.push_back(each);
                    }
                }
            }
        }
        catch(const c10::Error& error)
        {
            error_msg = error.what();
            tipl::out() << error_msg;
        }
        catch(...)
        {
            error_msg = "unknown error in training";
            tipl::out() << error_msg;
        }
        pause = true;
        aborted = true;
    };

    if(tipl::show_prog)
        validation_thread.reset(new std::thread(run_validation));
    else
    {
        run_validation();
        join();
    }
}

void train_unet::start(void)
{
    reading_status = augmentation_status = training_status = validation_status = "initializing";
    {
        stop();
        pause = false;
        aborted = false;
        running = true;
        error_msg.clear();
    }


    if(param.image_file_name.empty())
        return error_msg = "please specify the training data",aborted = true,void();


    while(model->errors.size() >= param.epoch*3)
    {
        tipl::out() << "prior training finished. restart training model...";
        model->prior_errors.insert(model->prior_errors.end(),model->errors.begin(),model->errors.begin() + param.epoch*3);
        model->errors.erase(model->errors.begin(),model->errors.begin() + param.epoch*3);
    }
    tipl::out() << "starting epoch: " << (cur_validation_epoch = cur_epoch = model->errors.size()/3);


    {
        model->to(param.device);
        model->train();
        if(!model->optimizer.get())
        {
            model->create_optimizer(param.learning_rate);
            if(std::filesystem::exists(model_path+".opt"))
            {
                tipl::out() << "loading existing optimizer " << model_path+".opt";
                try
                {
                    torch::load(*(model->optimizer),
                                (std::filesystem::path(model_path) += ".opt").make_preferred().string());
                }
                catch(const c10::Error& e)
                {
                    return tipl::error() << (error_msg = std::string("cannot load optimizer: ") + e.what()),aborted = true,void();
                }
            }
        }
    }

    {
        tipl::out() << "gpu count: " << torch::cuda::device_count();
        other_models.clear();
        for(int i = 1,gpu_count = torch::cuda::device_count();i<gpu_count;++i)
        {
            tipl::out() << "model added at cuda:" << i << std::endl;
            auto new_model = UNet3d(model->in_count,model->out_count,model->architecture);
            new_model->to(torch::Device(torch::kCUDA,i));
            new_model->train();
            other_models.push_back(new_model);
        }

        output_model = UNet3d(model->in_count,model->out_count,model->architecture);
        output_model->to(param.test_device);
        output_model->copy_from(*model);
        output_model->errors = model->errors;
    }



    {
        thread_count = po.get("thread_count",std::min<int>(8,std::thread::hardware_concurrency()));

        train_image = std::vector<tipl::image<3>>(param.image_file_name.size());
        train_label = std::vector<tipl::image<3>>(param.image_file_name.size());


        in_data_read_id = std::vector<size_t>(thread_count);
        in_file_read_id = std::vector<size_t>(thread_count);
        in_file_seed = std::vector<size_t>(thread_count);
        out_data = std::vector<tipl::image<3>>(thread_count);
        in_data = std::vector<tipl::image<3>>(thread_count);
        in_file = std::vector<tipl::image<3>>(thread_count);
        out_file = std::vector<tipl::image<3>>(thread_count);
        data_ready = std::vector<char>(thread_count,false);
        file_ready = std::vector<char>(thread_count,false);
        test_data_ready = false;
        test_in_tensor.clear();
        test_out_tensor.clear();
    }

    tipl::progress p("starting training");

    read_file();
    train();
    validate();
}

void train_unet::join(void)
{
    if(read_images.get())
    {
        read_images->join();
        read_images.reset();
    }
    if(augmentation_thread.get())
    {
        augmentation_thread->join();
        augmentation_thread.reset();
    }
    if(train_thread.get())
    {
        train_thread->join();
        train_thread.reset();
    }
    if(validation_thread.get())
    {
        validation_thread->join();
        validation_thread.reset();
    }
}

void train_unet::stop(void)
{
    pause = true;
    aborted = true;
    join();
}


std::string get_model_path(void)
{
    std::string model_path = po.get("model");
    if(!tipl::ends_with(model_path,"nz"))
        model_path += ".nz";
    if(!std::filesystem::exists(model_path) && std::filesystem::exists(po.exec_path+"/unet/"+model_path))
        po.set("model",model_path = po.exec_path+"/unet/"+model_path);
    return model_path;
}

std::string default_feature(int out_count)
{
    auto out = "conv" + std::to_string(out_count) + ",ks1,stride1";
    return
            "conv16,ks3,stride1+norm,leaky_relu+conv16,ks3,stride1+norm,leaky_relu\n"
            "conv32,ks3,stride2+norm,leaky_relu+conv32,ks3,stride1+norm,leaky_relu\n"
            "conv64,ks3,stride2+norm,leaky_relu+conv64,ks3,stride1+norm,leaky_relu\n"
            "conv128,ks3,stride2+norm,leaky_relu+conv128,ks3,stride1+norm,leaky_relu\n"
            "conv256,ks3,stride2+norm,leaky_relu+conv256,ks3,stride1+norm,leaky_relu\n"
            "conv256,ks3,stride2+norm,leaky_relu+conv256,ks3,stride1+norm,leaky_relu+conv_trans256,ks2,stride2\n"
            "conv256,ks3,stride1+norm,leaky_relu+conv256,ks3,stride1+norm,leaky_relu+" + out + "+conv_trans128,ks2,stride2\n" +
            "conv128,ks3,stride1+norm,leaky_relu+conv128,ks3,stride1+norm,leaky_relu+" + out + "+conv_trans64,ks2,stride2\n" +
            "conv64,ks3,stride1+norm,leaky_relu+conv64,ks3,stride1+norm,leaky_relu+" + out + "+conv_trans32,ks2,stride2\n" +
            "conv32,ks3,stride1+norm,leaky_relu+conv32,ks3,stride1+norm,leaky_relu+" + out + "+conv_trans16,ks2,stride2\n" +
            "conv16,ks3,stride1+norm,leaky_relu+conv16,ks3,stride1+norm,leaky_relu+" + out;
}


int tra(void)
{
    static train_unet train;
    if(train.running)
    {
        tipl::out() << "terminating training...";
        train.stop();
    }

    auto def_device = torch::hasCUDA()?"cuda:0":(torch::hasHIP()?"hip:0":(torch::hasMPS()?"mps:0":"cpu"));
    train.param.batch_size =        po.get("batch_size",train.param.batch_size);
    train.param.learning_rate =     po.get("learning_rate",train.param.learning_rate);
    train.param.epoch =             po.get("epoch",train.param.epoch);
    train.param.is_label =          po.get("is_label",train.param.is_label?1:0);
    train.param.cost_ce =           po.get("cost_ce",train.param.cost_ce ? 1:0);
    train.param.cost_dice =         po.get("cost_dice",train.param.cost_dice ? 1:0);
    train.param.cost_mse =          po.get("cost_mse",train.param.cost_mse ? 1:0);
    train.param.seed =              po.get("seed",((train.model->prior_errors.size()+train.model->errors.size())/3)/train.param.epoch);
    train.param.device = torch::Device(po.get("device",def_device));
    tipl::progress p("start training");

    {
        train.param.image_file_name = po.get_files("source");
        train.param.label_file_name = po.get_files("label");
        if(train.param.image_file_name.empty()||train.param.label_file_name.empty())
            return tipl::error() << "please specify training data using --source and --label",1;

        if(train.param.image_file_name.size()!=train.param.label_file_name.size())
            return tipl::error() << "different number of files found for image and label",1;

        for(size_t i = 0,sz = train.param.image_file_name.size();i<sz;++i)
            tipl::out() << std::filesystem::path(train.param.image_file_name[i]).filename().string() <<
                           "=>" << std::filesystem::path(train.param.label_file_name[i]).filename().string();
    }

    {
        tipl::progress prog("setting up model");
        train.model_path = get_model_path();
        if(std::filesystem::exists(train.model_path))
        {
            tipl::out() << "loading existing model " << train.model_path;
            if(!load_from_file(train.model,train.model_path.c_str()))
                return tipl::error() << "failed to load model from " << train.model_path,1;
        }
        else
        {
            tipl::image<3,char> I;
            tipl::shape<3> dim;
            tipl::vector<3> vs;

            if(!(tipl::io::gz_nifti(train.param.label_file_name[0],std::ios::in) >> I >>
                  [&](const auto& e){tipl::error() << "cannot load label file: " << e;}) ||
                !(tipl::io::gz_nifti(train.param.image_file_name[0],std::ios::in) >> dim >> vs >>
                  [&](const auto& e){tipl::error() << "cannot load image file: " << e;}))
                return 1;

            size_t in_count = po.get("in_count",1);
            size_t out_count = po.get("out_count",tipl::max_value(I)+1);
            std::string architecture = po.get("architecture",default_feature(out_count));

            try
            {
                train.model = UNet3d(in_count,out_count,architecture);
                tipl::out() << "dim: " << (train.model->dim = tipl::ml3d::round_up_size(tipl::v(32,32,32),dim));
                tipl::out() << "vs: " << (train.model->voxel_size = vs);
            }
            catch(...)
            {
                return tipl::error() << "invalid network structure ",1;
            }
        }
    }

    if(po.has("label_weight"))
        train.param.set_weight(po.get("label_weight"));

    {
        tipl::progress prog("visual augmentation options");
        QFile data(":/options.txt");
        if(!data.open(QIODevice::ReadOnly|QIODevice::Text))
            return tipl::error() << "cannot load options",1;

        QTextStream in(&data);
        QString last_root;
        while(!in.atEnd())
        {
            QStringList list = in.readLine().split('/');
            if(list.size()<5)
                continue;
            train.param.options[list[2].toUtf8().constData()] = po.get(list[2].toUtf8().constData(),list[4].toFloat());
        }
    }

    train.save_model_during_training = true;
    train.start();

    if(!train.error_msg.empty())
        return tipl::error() << train.error_msg,1;

    tipl::out() << "save model to " << train.model_path;
    if(!save_to_file(train.model,train.model_path.c_str()))
        return tipl::error() << "failed to save network to " << train.model_path,1;
    return 0;
}
