#include <QFile>
#include <QTextStream>
#include "train.hpp"
extern tipl::program_option<tipl::out> po;
bool load_from_file(UNet3d& model,const char* file_name);
bool save_to_file(UNet3d& model,const char* file_name);
using namespace std::chrono_literals;


bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          size_t in_count,
                          tipl::image<3>& input,
                          tipl::image<3>& label,
                          tipl::shape<3>& image_shape)
{
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    {
        tipl::io::gz_nifti nii(image_name,std::ios::in);
        if(!nii || nii.dim(4) != in_count)
            return false;
        nii.get_image_dimension(image_shape);
        if(image_shape.size() > 256*256*196)
            return false;
        input.resize(tipl::shape<3>(image_shape.width(),image_shape.height(),image_shape.depth()*in_count));
        for(int c = 0;c < in_count;++c)
        {
            auto I = input.alias(image_shape.size()*c,image_shape);
            nii >> I;
        }
        nii >> image_t;
    }

    tipl::io::gz_nifti nii(label_name,std::ios::in);
    if(!nii)
        return false;
    label.clear();
    label.resize(image_shape);
    if(nii.dim(4) > 1)
    {
        for(size_t index = 1;index <= nii.dim(4);++index)
        {
            tipl::image<3> I(image_shape);
            nii.to_space(I,image_t);
            for(size_t pos = 0;pos < I.size();++pos)
                if(I[pos])
                    label[pos] = index;
        }
    }
    else
        nii.to_space(label,image_t);
    return true;
}


void preprocessing(tipl::image<3>& image,tipl::image<3>& label,tipl::shape<3> from_dim,tipl::shape<3> to_dim)
{
    if(from_dim != to_dim)
    {
        auto shift = tipl::vector<3,int>(to_dim)-tipl::vector<3,int>(from_dim);
        shift /= 2;
        tipl::image<3> new_label(to_dim);
        tipl::draw(label,new_label,shift);
        new_label.swap(label);

        int in_count = image.depth()/from_dim[2];
        tipl::image<3> new_image(to_dim.multiply(tipl::shape<3>::z,in_count));
        for(int c = 0;c < in_count;++c)
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

    train_image = std::vector<tipl::image<3> >(param.image_file_name.size());
    train_label = std::vector<tipl::image<3> >(param.image_file_name.size());
    train_image_is_template = std::vector<bool>(param.image_file_name.size(),true);

    in_data_read_id = in_file_read_id = in_file_seed = std::vector<size_t>(thread_count);
    out_data = in_data = in_file = out_file = std::vector<tipl::image<3> >(thread_count);
    data_ready = file_ready = std::vector<bool>(thread_count,false);

    test_data_ready = false;
    test_in_tensor.clear();
    test_out_tensor.clear();

    // read training data
    read_images.reset(new std::thread([=]()
    {
        // checking template or subject images
        {
            for(size_t i = 0;i < param.image_file_name.size();++i)
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
                    tipl::out() << "template found: " << param.image_file_name[i];
            }
        }

        std::vector<size_t> template_indices;
        std::vector<size_t> non_template_indices;
        for(size_t i = 0;i < train_image_is_template.size();++i)
            if(train_image_is_template[i])
                template_indices.insert(template_indices.end(),i);
            else
                non_template_indices.insert(non_template_indices.end(),i);



        tipl::out() << "a total of " << param.image_file_name.size() << " training dataset" << std::endl;
        tipl::out() << "a total of " << param.test_image_file_name.size() << " testing dataset" << std::endl;

        //prepare test data
        {
            for(int read_id = 0;read_id < param.test_image_file_name.size() && !aborted;++read_id)
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
                if(!read_image_and_label(param.test_image_file_name[read_id],
                                         param.test_label_file_name[read_id],
                                         model->in_count,
                                         input_image,input_label,input_shape))
                {
                    error_msg = "cannot read image or label data for ";
                    error_msg += std::filesystem::path(param.test_image_file_name[read_id]).filename().string();
                    aborted = true;
                    return;
                }

                preprocessing(input_image,input_label,input_shape,model->dim);


                if(model->out_count > 1)
                    tipl::expand_label_to_dimension(input_label,model->out_count,false);
                else
                    tipl::normalize(input_label);

                try{
                test_in_tensor.push_back(torch::from_blob(input_image.data(),
                    {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_tensor.push_back(torch::from_blob(input_label.data(),
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                }
                catch(const c10::Error& error)
                {
                    error_msg = "test tensor allocation error: ";
                    error_msg += error.what();
                    aborted = true;
                    return;
                }

            }
            test_data_ready = true;
        }
        // prepare training data

        size_t seed = model->total_training_count;
        std::mt19937 gen(0);
        std::uniform_int_distribution<int> template_gen(0, std::max<int>(1,template_indices.size())-1);
        std::uniform_int_distribution<int> non_template_gen(0, std::max<int>(1,non_template_indices.size())-1);

        for(size_t thread = 0;!aborted;++thread,++seed)
        {
            if(thread >= in_file.size())
                thread = 0;

            size_t read_id =
            (non_template_indices.empty() || seed % param.batch_size < template_indices.size()) ?
                template_indices[template_gen(gen)] : non_template_indices[non_template_gen(gen)];

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
                while(!file_ready[thread] || pause)
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
                if(model->out_count > 1)
                    tipl::expand_label_to_dimension(out_data_thread,model->out_count,false);

                while(data_ready[thread] || pause)
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
    for(size_t i = 0;i < file_ready.size();++i)
    {
        s1[i] = file_ready[i] ? '-' : '_';
        s2[i] = data_ready[i] ? '-' : '_';
    }
    return s1 + "|" + s2;
}

void train_unet::train(void)
{
    auto calc_losses = [](torch::Tensor& pred_raw, torch::Tensor& target_all, int C)
        -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    {
        // 1. Probabilistic Clamping for ReLU outputs
        auto pred_clamped = pred_raw.clamp(0.0, 1.0);

        // 2. Scaled Categorical Cross Entropy
        // Normalize by log(C) so a random guess is always ~1.0
        auto ce = -torch::mean(torch::sum(target_all * torch::log(pred_clamped + 1e-7), 1));
        float ce_normalization = std::log(static_cast<float>(C));
        ce /= (ce_normalization > 0.5f ? ce_normalization : 1.0f);

        // 3. Dice Loss (Overlap-based)
        // Inherently normalized between 0 and 1
        auto dims = std::vector<int64_t>{2, 3, 4};
        auto intersection = torch::sum(pred_clamped * target_all, dims);
        auto cardinality = torch::sum(pred_clamped + target_all, dims);
        auto dice = 1.0f - torch::mean((2.f * intersection + 1e-5) / (cardinality + 1e-5));

        // 4. Scaled MSE
        // Multiply by C to balance the average-over-channels denominator
        auto mse = torch::mse_loss(pred_raw, target_all) * static_cast<float>(C);

        return {ce, dice, mse};
    };

    auto run_training = [=](){
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false; }
        } guard(running);

        try{

            std::shared_ptr<torch::optim::Optimizer> optimizer(
                        new torch::optim::Adam(model->parameters(),
                            torch::optim::AdamOptions(param.learning_rate)));
            float best_val_error = std::numeric_limits<float>::max();
            int patience_counter = 0;
            int max_patience = 8;
            const float decay_factor = 0.5f;
            const float min_lr = 1e-7f;

            while(!test_data_ready || pause)
            {
                std::this_thread::sleep_for(100ms);
                if(aborted)
                    return;
            }

            auto to_chart = [](float error)->int{return int(std::max<float>(0.0f,std::min<float>(79.0f,(-std::log10(error))*80.0f/3.0f)));};
            for (size_t cur_data_index = 0; cur_epoch < param.epoch && !aborted;cur_epoch++,cur_data_index += param.batch_size)
            {
                if(param.learning_rate != optimizer->defaults().get_lr())
                {
                    tipl::out() << "set learning rate to " << param.learning_rate << std::endl;
                    optimizer->defaults().set_lr(param.learning_rate);
                }

                training_status = "training ";
                size_t model_count = std::min<int>(1 + other_models.size(),param.batch_size);
                for(auto& each : other_models)
                    each->copy_from(*model);
                std::mutex m;
                size_t b = 0;
                std::vector<float> error_each_model(model_count);
                tipl::par_for(model_count,[&](size_t model_id)
                {
                    try{

                        auto& cur_model = (model_id == 0 ? model:other_models[model_id-1]);
                        while(!aborted)
                        {
                            size_t thread = 0;
                            {
                                std::lock_guard<std::mutex> lock(m);
                                if(b >= param.batch_size)
                                    break;
                                thread = (cur_data_index+(b++))%data_ready.size();
                                training_status += std::to_string(in_data_read_id[thread]) + "." + std::to_string(model_id) +
                                                   (train_image_is_template[in_data_read_id[thread]] ? "t":"s");
                            }

                            while(!data_ready[thread] || pause)
                            {
                                std::this_thread::sleep_for(100ms);
                                if(aborted)
                                    return;
                            }

                            torch::Tensor target_probs = torch::from_blob(out_data[thread].data(),
                                    {1,cur_model->out_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(cur_model->device());
                            torch::Tensor in_tensor = torch::from_blob(in_data[thread].data(),
                                    {1,cur_model->in_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(cur_model->device());
                            data_ready[thread] = false;

                            auto outputs = cur_model->forward(in_tensor);

                            torch::Tensor total_loss;
                            {
                                for(size_t k = 0; k < outputs.size(); ++k)
                                {
                                    // 1. Handle Downsampling for Target
                                    torch::Tensor cur_target_probs;
                                    if(k == 0)
                                        cur_target_probs = target_probs;
                                    else
                                        cur_target_probs = torch::nn::functional::interpolate(target_probs,
                                            torch::nn::functional::InterpolateFuncOptions()
                                            .size(std::vector<int64_t>{outputs[k].size(2), outputs[k].size(3), outputs[k].size(4)})
                                            .mode(torch::kTrilinear).align_corners(false));

                                    auto [ce, dice, mse] = calc_losses(outputs[k], cur_target_probs,cur_model->out_count);
                                    float level_weight = 1.0f / (1 << k);
                                    auto level_loss = (ce + dice + mse) * level_weight;

                                    if(total_loss.defined()) total_loss += level_loss;
                                    else total_loss = level_loss;
                                }
                            };
                            error_each_model[model_id] = total_loss.item().toFloat();
                            total_loss.backward();
                        }
                    }
                    catch(const c10::Error& error)
                    {
                        tipl::out() << (error_msg = std::string("during ") + training_status + ":" + error.what());
                        aborted = true;
                        return;
                    }

                },thread_count);

                if(aborted)
                    return;
                {
                    training_status = "update model";
                    for(auto& each : other_models)
                        model->add_gradient_from(*each);
                    optimizer->step();
                    optimizer->zero_grad();
                    model->total_training_count += param.batch_size;

                }

                std::vector<float> errors;
                if(!test_in_tensor.empty())
                {
                    test_model->copy_from(*model);
                    test_model->eval();
                    torch::NoGradGuard no_grad; // Disable gradient for validation
                    float cur_error = 0.0f;
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                    {
                        float ce_v,dice_v,mse_v;
                        auto [ce, dice, mse] = calc_losses(test_model->forward(test_in_tensor[i])[0], test_out_tensor[i],test_model->out_count);
                        errors.push_back(ce_v = ce.item().toFloat());
                        errors.push_back(dice_v = dice.item().toFloat());
                        errors.push_back(mse_v = mse.item().toFloat());
                        cur_error += ce_v + dice_v + mse_v;
                    }
                    if(test_error.empty())
                    {
                        test_error.resize(errors.size());
                        test_error_name = {"ce","dice","mse"};
                    }
                    for(size_t i = 0;i < errors.size();++i)
                        test_error[i].push_back(errors[i]);

                    if (cur_error < best_val_error * 0.999f) // 0.1% improvement threshold
                    {
                        best_val_error = cur_error;
                        patience_counter = 0;
                        std::scoped_lock<std::mutex> lock(output_model_mutex);
                        output_model->copy_from(*model);
                    }
                    else
                    {
                        patience_counter++;
                        if (patience_counter >= max_patience && param.learning_rate > min_lr)
                        {
                            param.learning_rate *= decay_factor;
                            patience_counter = 0;
                            max_patience *= 2;
                        }
                    }
                }


                {
                    if(!cur_epoch)
                        tipl::out()     << "1                        0.1                        0.01                   0.001";
                    std::string out = cur_epoch % 100 ?
                            "|                         |                          |                         |":
                            "|-------------------------|--------------------------|-------------------------|";
                    if(!errors.empty())
                    {
                        out[to_chart(errors[0])] = 'C';
                        out[to_chart(errors[1])] = 'D';
                        out[to_chart(errors[2])] = 'M';
                    }

                    std::string epoch_string = "|" + std::to_string(cur_epoch);
                    std::copy(epoch_string.begin(),epoch_string.end(),out.begin());
                    tipl::out() << out;

                }

                if(po.has("network") &&
                   (((cur_epoch + 1) % 500 == 0) || cur_epoch+1 == param.epoch))
                {
                    if(!save_to_file(model,po.get("network").c_str()))
                    {
                        error_msg = "failed to save network";
                        aborted = true;
                    }
                    if(!save_error_to(po.get("error",po.get("network") + ".error.txt").c_str()))
                        error_msg = "failed to save error";
                }
            }
            tipl::out() << (training_status = "training completed");

        }
        catch(const c10::Error& error)
        {
            tipl::out() << (error_msg = std::string("during ") + training_status + ":" + error.what());
        }
        catch(...)
        {
            tipl::out() << (error_msg = "unknown error in training");
        }
        pause = aborted = true;
    };

    if(tipl::show_prog)
        train_thread.reset(new std::thread(run_training));
    else
    {
        run_training();
        join();
    }
}
void train_unet::start(void)
{
    tipl::progress p("starting training");
    // reset
    reading_status = augmentation_status = training_status = "initializing";
    {
        stop();
        pause = aborted = false;
        running = true;
        error_msg.clear();
        test_error.clear();
        test_error_name.clear();
        cur_epoch = 0;
    }

    if(param.image_file_name.empty())
    {
        error_msg = "please specify the training data";
        aborted = true;
        return;
    }

    if(model->total_training_count == 0)
    {
        tipl::io::gz_nifti in(param.image_file_name[0],std::ios::in);
        if(!in)
        {
            error_msg = in.error_msg;
            aborted = true;
            return;
        }
        in.toLPS();
        in.get_image_dimension(model->dim);
        in.get_voxel_size(model->voxel_size);
        model->dim = tipl::ml3d::round_up_size(model->dim);
        model->voxel_size = model->voxel_size[0];
        tipl::out() << "set network input sizes: " << model->dim << " with resolution:" << model->voxel_size <<std::endl;
    }

    model->to(param.device);
    model->train();
    output_model = UNet3d(model->in_count,model->out_count,model->feature_string);
    output_model->to(param.device);
    output_model->copy_from(*model);

    if(!param.test_image_file_name.empty())
    {
        test_model = UNet3d(model->in_count,model->out_count,model->feature_string);
        test_model->to(param.device);
    }

    tipl::out() << "gpu count: " << torch::cuda::device_count();

    other_models.clear();
    for(size_t i = 1;i < torch::cuda::device_count();++i)
    {
        tipl::out() << "model added at cuda:" << (i % torch::cuda::device_count());
        other_models.push_back(UNet3d(model->in_count,model->out_count,model->feature_string));
        other_models.back()->to(torch::Device(torch::kCUDA,i % torch::cuda::device_count()));
        other_models.back()->train();
    }

    read_file();
    train();        
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
}
void train_unet::stop(void)
{
    pause = aborted = true;
    join();
}
bool train_unet::save_error_to(const char* file_name)
{
    std::ofstream out(file_name);
    if(!out || test_error.empty())
        return false;
    out << "epoch\t" << tipl::merge(test_error_name,'\t') << std::endl;
    for(size_t i = 0;i < test_error[0].size();++i)
    {
        out << i << "\t";
        for(size_t j = 0;j < test_error.size();++j)
            out << test_error[j][i];
        out << std::endl;
    }
    return true;
}

bool get_label_info(const std::string& label_name,std::vector<int>& out_count,bool& is_label);


std::string get_network_path(void)
{
    std::string network = po.get("network");
    if(!tipl::ends_with(network,"net.gz"))
        network += ".net.gz";
    if(!std::filesystem::exists(network) && std::filesystem::exists(po.exec_path + "/network/" + network))
        po.set("network",network = po.exec_path + "/network/" + network);
    return network;
}
int tra(void)
{
    static train_unet train;
    if(train.running)
    {
        tipl::out() << "terminating training...";
        train.stop();
    }

    tipl::progress p("start training");

    // loading training data
    {
        train.param.image_file_name.clear();
        train.param.label_file_name.clear();
        if(!po.get_files("source",train.param.image_file_name) ||
           !po.get_files("label",train.param.label_file_name))
        {
            tipl::error() << po.error_msg;
            return 1;
        }

        if(train.param.image_file_name.size() != train.param.label_file_name.size())
        {
            tipl::error() << "different number of files found for image and label";
            return 1;
        }
        if(train.param.image_file_name.empty())
        {
            tipl::error() << "no available training images";
            return 1;
        }
        for(size_t i = 0;i < train.param.image_file_name.size();++i)
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
        if(po.get("out_count",train.model->out_count) != train.model->out_count)
        {
            tipl::out() << "changing output channel" << std::endl;
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
        std::string feature_string = po.get("feature_string",
                out_count <= 6 ? "8x8+16x16+32x32+64x64+128x128" :
                (out_count <= 10 ? "16x16+32x32+64x64+128x128+128x128" : "32x32+64x64+128x128+128x128+256x256"));
        try{
            tipl::out() << "create new network with structure " << feature_string;
            train.model = UNet3d(in_count,out_count,feature_string);
        }
        catch(...)
        {
            tipl::error() << "invalid network structure ";
            return 1;
        }
    }


    train.param.batch_size = po.get("batch_size",train.param.batch_size);
    train.param.learning_rate = po.get("learning_rate",train.param.learning_rate);
    train.param.epoch = po.get("epoch",train.param.epoch);

    train.param.is_label = po.get("is_label",train.param.is_label ? 1:0);
    train.param.device = torch::Device(po.get("device",torch::hasCUDA() ? "cuda:0" :
                                                       (torch::hasHIP() ? "hip:0" :
                                                       (torch::hasMPS() ? "mps:0": "cpu"))));
    if(po.has("label_weight"))
        train.param.set_weight(po.get("label_weight"));


    // setting up the parameters for visual perception augmentation
    {
        QFile data(":/options.txt");
        if (!data.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            tipl::error() << "cannot load options";
            return 1;
        }
        QTextStream in(&data);
        QString last_root;
        while (!in.atEnd())
        {
            QStringList list = in.readLine().split('/');
            if(list.size() < 5)
                continue;
            train.param.options[list[2].toStdString()] = po.get(list[2].toStdString().c_str(),list[4].toFloat());
        }
    }


    train.start();

    if(!train.error_msg.empty())
    {
        tipl::error() << train.error_msg;
        return 1;
    }
    return 0;
}
