#include <QFile>
#include <QTextStream>
#include "train.hpp"
extern tipl::program_option<tipl::out> po;
using namespace std::chrono_literals;
bool load_from_file(UNet3d& model,const char* file_name);
bool save_to_file(UNet3d& model,const char* file_name);

bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          size_t in_count,
                          tipl::image<3>& input,
                          tipl::image<3>& label,
                          tipl::shape<3>& image_shape)
{
    std::scoped_lock<std::mutex> lock(tipl::io::nifti_do_not_show_process);
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    {
        tipl::io::gz_nifti nii(image_name,std::ios::in);
        if(!nii)
        {
            tipl::error() << nii.error_msg;
            return false;
        }
        if(nii.dim(4)!=in_count)
        {
            tipl::error() << "image channel number does not match model input channel";
            return false;
        }
        nii.get_image_dimension(image_shape);
        input.resize(tipl::shape<3>(image_shape.width(),image_shape.height(),image_shape.depth()*in_count));
        for(int c=0;c<in_count;++c)
        {
            auto I = input.alias(image_shape.size()*c,image_shape);
            nii >> I;
        }
        nii >> image_t;
    }

    tipl::io::gz_nifti nii(label_name,std::ios::in);
    if(!nii)
    {
        tipl::error() << nii.error_msg;
        return false;
    }
    label.clear();
    label.resize(image_shape);
    if(nii.dim(4)>1)
    {
        for(size_t index=1;index<=nii.dim(4);++index)
        {
            tipl::image<3> I(image_shape);
            nii.to_space<tipl::majority>(I,image_t);
            for(size_t pos=0,sz=I.size();pos<sz;++pos)
                if(I[pos])
                    label[pos] = index;
        }
    }
    else
        nii.to_space<tipl::majority>(label,image_t);
    return true;
}

void preprocessing(tipl::image<3>& image,tipl::image<3>& label,tipl::shape<3> from_dim,tipl::shape<3> to_dim)
{
    if(from_dim!=to_dim)
    {
        auto shift = tipl::vector<3,int>(to_dim)-tipl::vector<3,int>(from_dim);
        shift /= 2;
        tipl::image<3> new_label(to_dim);
        tipl::draw(label,new_label,shift);
        new_label.swap(label);

        int in_count = image.depth()/from_dim[2];
        tipl::image<3> new_image(to_dim.multiply(tipl::shape<3>::z,in_count));
        for(int c=0;c<in_count;++c)
        {
            auto from = image.alias(c*from_dim.size(),from_dim);
            auto to = new_image.alias(c*to_dim.size(),to_dim);
            tipl::draw(from,to,shift);
            tipl::normalize(to);
        }
        new_image.swap(image);
    }
}

void train_unet::read_file(void)
{
    thread_count = po.get("thread_count",std::min<int>(8,std::thread::hardware_concurrency()));

    train_image = std::vector<tipl::image<3>>(param.image_file_name.size());
    train_label = std::vector<tipl::image<3>>(param.image_file_name.size());
    train_image_is_template = std::vector<char>(param.image_file_name.size(),false);

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

    read_images.reset(new std::thread([=]()
    {
        for(size_t i=0;i<param.image_file_name.size();++i)
        {
            reading_status = "checking ";
            reading_status += param.image_file_name[i];
            tipl::io::gz_nifti in(param.image_file_name[i].c_str(),std::ios::in);
            if(!in)
            {
                error_msg = in.error_msg;
                aborted = true;
                return;
            }
            train_image_is_template[i] = in.is_mni();
            if(train_image_is_template[i])
            {
                tipl::out() << "template found: " << param.image_file_name[i];
                param.test_image_file_name.push_back(param.image_file_name[i]);
                param.test_label_file_name.push_back(param.label_file_name[i]);
            }
        }

        std::vector<size_t> template_indices;
        std::vector<size_t> non_template_indices;
        for(size_t i=0;i<train_image_is_template.size();++i)
            if(train_image_is_template[i])
                template_indices.push_back(i);
            else
                non_template_indices.push_back(i);

        tipl::out() << "a total of " << param.image_file_name.size() << " training dataset\n";
        tipl::out() << "a total of " << param.test_image_file_name.size() << " testing dataset\n";

        for(int read_id=0;read_id<param.test_image_file_name.size()&&!aborted;++read_id)
        {
            while(pause)
            {
                std::this_thread::sleep_for(100ms);
                if(aborted)
                    return;
            }

            reading_status = "reading ";
            reading_status += std::filesystem::path(param.test_image_file_name[read_id]).filename().string();
            reading_status += " and ";
            reading_status += std::filesystem::path(param.test_label_file_name[read_id]).filename().string();

            tipl::image<3> input_image,input_label;
            tipl::shape<3> input_shape;
            if(!read_image_and_label(param.test_image_file_name[read_id],param.test_label_file_name[read_id],model->in_count,input_image,input_label,input_shape))
            {
                error_msg = "cannot read image or label data for ";
                error_msg += std::filesystem::path(param.test_image_file_name[read_id]).filename().string();
                aborted = true;
                return;
            }

            preprocessing(input_image,input_label,input_shape,model->dim);

            if(model->out_count>1)
                tipl::expand_label_to_dimension(input_label,model->out_count,false);
            else
                tipl::normalize(input_label);

            try
            {
                test_in_tensor.push_back(torch::from_blob(input_image.data(),{1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device,true));
                test_out_tensor.push_back(torch::from_blob(input_label.data(),{1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device,true));
            }
            catch(const c10::Error& error)
            {
                error_msg = std::string("test tensor allocation error: ")+error.what();
                aborted = true;
                return;
            }
        }
        test_data_ready = true;

        size_t seed = param.batch_size*(model->errors.size()/3);
        std::mt19937 gen(0);
        std::uniform_int_distribution<int> template_gen(0,std::max<int>(1,template_indices.size())-1);
        std::uniform_int_distribution<int> non_template_gen(0,std::max<int>(1,non_template_indices.size())-1);

        for(size_t thread=0;!aborted;++thread,++seed)
        {
            if(thread>=in_file.size())
                thread = 0;

            size_t read_id = (non_template_indices.empty()||seed%param.batch_size<template_indices.size()) ? template_indices[template_gen(gen)] : non_template_indices[non_template_gen(gen)];
            tipl::image<3> image,label;

            if(train_image[read_id].empty())
            {
                reading_status = "reading ";
                reading_status += std::filesystem::path(param.image_file_name[read_id]).filename().string();
                reading_status += " and ";
                reading_status += std::filesystem::path(param.label_file_name[read_id]).filename().string();
                tipl::shape<3> image_shape;
                if(!read_image_and_label(param.image_file_name[read_id],param.label_file_name[read_id],model->in_count,image,label,image_shape))
                {
                    error_msg = "cannot read image or label data for ";
                    error_msg += std::filesystem::path(param.image_file_name[read_id]).filename().string();
                    aborted = true;
                    return;
                }
                reading_status = "preprocessing";
                preprocessing(image,label,image_shape,model->dim);
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
            {
                std::this_thread::sleep_for(100ms);
                if(aborted)
                    return;
            }
            in_file[thread].swap(image);
            out_file[thread].swap(label);
            in_file_seed[thread] = seed;
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
                {
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }
                tipl::image<3> in_data_thread,out_data_thread;
                size_t read_id = in_file_read_id[thread];
                size_t seed = in_file_seed[thread];
                in_data_thread.swap(in_file[thread]);
                out_data_thread.swap(out_file[thread]);
                file_ready[thread] = false;

                {
                    std::lock_guard<std::mutex> lock(m);
                    augmentation_status = "augmenting ";
                    augmentation_status += std::filesystem::path(param.image_file_name[read_id]).filename().string();
                }

                visual_perception_augmentation(param.options,in_data_thread,out_data_thread,param.is_label,model->dim,seed);
                if(model->out_count>1)
                    tipl::expand_label_to_dimension(out_data_thread,model->out_count,false);

                while(data_ready[thread]||pause)
                {
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }
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
    for(size_t i=0;i<file_ready.size();++i)
    {
        s1[i] = file_ready[i] ? '-' : '_';
        s2[i] = data_ready[i] ? '-' : '_';
    }
    return s1+"|"+s2;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> calc_losses(const torch::Tensor& pred_raw,const torch::Tensor& target_tissues,int C)
{
    auto ce = torch::nn::functional::cross_entropy(pred_raw,target_tissues);
    auto pred_probs = torch::softmax(pred_raw,1);
    auto inter = torch::sum(pred_probs*target_tissues,{2,3,4});
    auto card = torch::sum(pred_probs+target_tissues,{2,3,4});
    auto dice = 1.0f-torch::mean((2.f*inter+1e-5)/(card+1e-5));
    auto mse = torch::mse_loss(pred_probs,target_tissues)*static_cast<float>(C);
    return {ce,dice,mse};
}

void train_unet::train(void)
{
    auto run_training = [=]()
    {
        try
        {
            model->report += " Training was conducted over "+std::to_string(param.epoch)+" epochs ";
            model->report += "using a batch size of "+std::to_string(param.batch_size)+". ";
            model->report += "Optimization employed an initial learning rate of "+std::to_string(param.learning_rate)+" using SGD with Nesterov momentum.";

            std::vector<torch::Tensor> decay_params,no_decay_params;
            for(auto& p:model->named_parameters())
                if(p.key().find("bias")!=std::string::npos||p.key().find("norm")!=std::string::npos||p.key().find("bn")!=std::string::npos||p.key().find("out_conv")!=std::string::npos)
                    no_decay_params.push_back(p.value());
                else
                    decay_params.push_back(p.value());

            double base_wd = 3e-5;
            std::vector<torch::optim::OptimizerParamGroup> groups;

            auto opt_d = std::make_unique<torch::optim::SGDOptions>(param.learning_rate);
            opt_d->momentum(0.99);
            opt_d->nesterov(true);
            opt_d->weight_decay(base_wd);

            auto opt_nd = std::make_unique<torch::optim::SGDOptions>(param.learning_rate);
            opt_nd->momentum(0.99);
            opt_nd->nesterov(true);
            opt_nd->weight_decay(0.0);

            groups.push_back(torch::optim::OptimizerParamGroup(decay_params,std::move(opt_d)));
            groups.push_back(torch::optim::OptimizerParamGroup(no_decay_params,std::move(opt_nd)));

            auto optimizer = std::make_shared<torch::optim::SGD>(groups,torch::optim::SGDOptions(param.learning_rate));

            if(po.has("network") && cur_epoch && std::filesystem::exists(po.get("network")+".opt"))
            {
                torch::load(*optimizer,po.get("network")+".opt");
                tipl::out() << "optimizer state found. training is resumed at epoch " << cur_epoch;
            }


            size_t cur_data_index = 0;
            for(;cur_epoch<param.epoch&&!aborted;++cur_epoch)
            {
                training_status = "training";
                double poly = std::pow(1.0-(double)cur_epoch/param.epoch,0.9);
                double cur_lr = param.learning_rate*poly;

                for(auto& group:optimizer->param_groups())
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

                tipl::par_for(active_threads,[&](size_t thread_id)
                {
                    auto cur_model = (thread_id==0) ? model : other_models[thread_id-1];
                    auto dev = cur_model->device();
                    torch::DeviceGuard guard(dev);
                    while(!aborted)
                    {
                        int b = next_batch_idx.fetch_add(1);
                        if(b>=param.batch_size)
                            break;
                        size_t data_idx = (cur_data_index+b)%data_ready.size();
                        while(!data_ready[data_idx]||pause)
                        {
                            std::this_thread::sleep_for(10ms);
                            if(aborted)
                                return;
                        }

                        auto target = torch::from_blob(out_data[data_idx].data(),{1,cur_model->out_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(dev,true);
                        auto in = torch::from_blob(in_data[data_idx].data(),{1,cur_model->in_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(dev,true);
                        data_ready[data_idx] = false;

                        auto outputs = cur_model->forward(in);
                        torch::Tensor total_loss,active_target = target;
                        for(size_t k=0;k<outputs.size();++k)
                        {
                            if(k>0)
                            {
                                std::vector<int64_t> target_size = {active_target.size(2)/2,active_target.size(3)/2,active_target.size(4)/2};
                                auto opt = torch::nn::functional::InterpolateFuncOptions().size(target_size).mode(torch::kTrilinear).align_corners(false);
                                active_target = torch::nn::functional::interpolate(active_target,opt);
                            }

                            auto [ce,dice,mse] = calc_losses(outputs[k],active_target,cur_model->out_count);
                            auto level_loss = (ce+dice)*(1.0f/(1<<k));

                            if(total_loss.defined())
                                total_loss += level_loss;
                            else
                                total_loss = level_loss;
                        }
                        total_loss.backward();
                    }
                },active_threads);

                cur_data_index += param.batch_size;

                if(aborted)
                    return;

                training_status = "update model";
                for(auto& each:other_models)
                    model->add_gradient_from(*each);

                optimizer->step();
                optimizer->zero_grad();

                while(cur_validation_epoch<cur_epoch||pause)
                {
                    std::this_thread::sleep_for(10ms);
                    if(aborted)
                        return;
                }
                std::scoped_lock<std::mutex> lock(output_model_mutex);
                output_model->copy_from(*model);

                if(po.has("network")&&(cur_epoch+1)%100==0)
                {
                    std::string net_path = po.get("network");
                    save_to_file(model,net_path.c_str());
                    torch::save(*optimizer,net_path+".opt");
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
            tipl::time t;
            for(;cur_validation_epoch<param.epoch&&!aborted;++cur_validation_epoch)
            {
                while(cur_epoch<=cur_validation_epoch||!test_data_ready||pause)
                {
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }
                std::vector<float> errors;
                if(!test_in_tensor.empty())
                {
                    torch::NoGradGuard no_grad;
                    output_model->eval();
                    for(size_t i=0;i<test_in_tensor.size();++i)
                    {
                        float ce_v,dice_v,mse_v;
                        auto [ce,dice,mse] = calc_losses(output_model->forward(test_in_tensor[i])[0],test_out_tensor[i],output_model->out_count);
                        errors.push_back(ce_v = ce.item().toFloat());
                        errors.push_back(dice_v = dice.item().toFloat());
                        errors.push_back(mse_v = mse.item().toFloat());
                    }
                    {
                        std::scoped_lock<std::mutex> lock(error_mutex);
                        for(size_t i=0;i<errors.size();++i)
                            model->errors.push_back(errors[i]);
                    }
                }
                {
                    if(!cur_validation_epoch)
                        tipl::out() << "1                        0.1                        0.01                   0.001";

                    if(cur_validation_epoch%100==0)
                    {
                        std::string out = "|-------------------------|--------------------------|-------------------------|";
                        auto str = t.to_string();
                        double cur_lr = param.learning_rate*std::pow(1.0-(double)cur_validation_epoch/param.epoch,0.9);
                        str += "-lr:"+std::to_string(cur_lr);
                        if(cur_validation_epoch>start_validation_epoch)
                        {
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()-start_time).count();
                            auto total_sec = elapsed*param.epoch/(cur_validation_epoch-start_validation_epoch);
                            str += " total:"+std::to_string(total_sec/3600)+"h"+std::to_string((total_sec%3600)/60)+"m";
                        }
                        size_t copy_len = std::min(str.length(),out.length()-2);
                        std::copy(str.begin(),str.begin()+copy_len,out.begin()+1);
                        tipl::out() << out;
                    }

                    std::string out = "|                         |                          |                         |";
                    if(!errors.empty())
                    {
                        auto to_chart = [](float error)->int{return int(std::max<float>(0.0f,std::min<float>(79.0f,(-std::log10(error))*80.0f/3.0f)));};
                        out[to_chart(errors[0])] = 'C';
                        out[to_chart(errors[1])] = 'D';
                        out[to_chart(errors[2])] = 'M';
                    }
                    tipl::out() << out << cur_validation_epoch;
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
    tipl::progress p("starting training");
    reading_status = augmentation_status = training_status = validation_status = "initializing";
    {
        stop();
        pause = false;
        aborted = false;
        running = true;
        error_msg.clear();
    }

    if(param.image_file_name.empty())
    {
        error_msg = "please specify the training data";
        aborted = true;
        return;
    }

    if(model->errors.empty() && !model->init_dimension(param.image_file_name[0]))
    {
        error_msg = model->error_msg;
        aborted = true;
        return;
    }

    cur_epoch = model->errors.size()/3;
    cur_validation_epoch = cur_epoch;

    model->to(param.device);
    model->train();

    other_models.clear();
    for(int i=1,gpu_count=torch::cuda::device_count();i<gpu_count;++i)
    {
        tipl::out() << "model added at cuda:" << i << std::endl;
        auto new_model = UNet3d(model->in_count,model->out_count,model->feature_string);
        new_model->to(torch::Device(torch::kCUDA,i));
        new_model->train();
        other_models.push_back(new_model);
    }

    output_model = UNet3d(model->in_count,model->out_count,model->feature_string);
    output_model->to(param.device);
    output_model->copy_from(*model);
    tipl::out() << "gpu count: " << torch::cuda::device_count();

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

bool get_label_info(const std::string& label_name,std::vector<int>& out_count,bool& is_label);

std::string get_network_path(void)
{
    std::string network = po.get("network");
    if(!tipl::ends_with(network,"nz"))
        network += ".nz";
    if(!std::filesystem::exists(network)&&std::filesystem::exists(po.exec_path+"/unet/"+network))
    {
        network = po.exec_path+"/unet/"+network;
        po.set("network",network);
    }
    return network;
}

std::string default_feature(int out_count)
{
    return "32x32+64x64+128x128+256x256+256x256";
}

int tra(void)
{
    static train_unet train;
    if(train.running)
    {
        tipl::out() << "terminating training...";
        train.stop();
    }

    train.param.batch_size = po.get("batch_size",train.param.batch_size);
    train.param.learning_rate = po.get("learning_rate",train.param.learning_rate);
    train.param.epoch = po.get("epoch",train.param.epoch);
    train.param.is_label = po.get("is_label",train.param.is_label ? 1 : 0);
    train.param.device = torch::Device(po.get("device",torch::hasCUDA() ? "cuda:0" : (torch::hasHIP() ? "hip:0" : (torch::hasMPS() ? "mps:0" : "cpu"))));


    tipl::progress p("start training");

    {
        train.param.image_file_name = po.get_files("source");
        train.param.label_file_name = po.get_files("label");

        if(train.param.image_file_name.empty()||train.param.label_file_name.empty())
        {
            tipl::error() << po.error_msg;
            return 1;
        }

        if(train.param.image_file_name.size()!=train.param.label_file_name.size())
        {
            tipl::error() << "different number of files found for image and label";
            return 1;
        }

        for(size_t i=0;i<train.param.image_file_name.size();++i)
            tipl::out() << std::filesystem::path(train.param.image_file_name[i]).filename().string() << "=>" << std::filesystem::path(train.param.label_file_name[i]).filename().string();
    }

    auto network = get_network_path();
    if(std::filesystem::exists(network))
    {
        tipl::out() << "loading existing network " << network;
        if(!load_from_file(train.model,network.c_str()))
        {
            tipl::error() << "failed to load model from " << network;
            return 1;
        }

        tipl::out() << train.model->get_info();


        if(po.get("out_count",train.model->out_count)!=train.model->out_count)
        {
            tipl::out() << "changing output channel\n";
            auto new_model = UNet3d(train.model->in_count,po.get("out_count",train.model->out_count),train.model->feature_string);
            new_model->copy_from(*train.model.get());
            train.model = new_model;
        }
    }
    else
    {
        std::vector<int> label_count;
        if(!get_label_info(train.param.label_file_name[0].c_str(),label_count,train.param.is_label))
        {
            tipl::error() << "cannot open the label file" << train.param.label_file_name[0];
            return 1;
        }
        size_t in_count = po.get("in_count",1);
        size_t out_count = po.get("out_count",label_count.size());
        std::string feature_string = po.get("feature_string",default_feature(out_count));
        try
        {
            tipl::out() << "create new network with structure " << feature_string;
            train.model = UNet3d(in_count,out_count,feature_string);
        }
        catch(...)
        {
            tipl::error() << "invalid network structure ";
            return 1;
        }
    }


    if(po.has("label_weight"))
        train.param.set_weight(po.get("label_weight"));

    {
        tipl::out() << "visual augmentation options";
        QFile data(":/options.txt");
        if(!data.open(QIODevice::ReadOnly|QIODevice::Text))
        {
            tipl::error() << "cannot load options";
            return 1;
        }
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

    train.start();

    if(!train.error_msg.empty())
    {
        tipl::error() << train.error_msg;
        return 1;
    }
    {
        tipl::out() << "save model to " << po.get("network","model.nz");
        if(!save_to_file(train.model,po.get("network").c_str()))
            tipl::error() << "failed to save network to " << po.get("network");
    }
    return 0;
}
