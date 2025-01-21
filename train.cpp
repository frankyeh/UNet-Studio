#include <QFile>
#include <QTextStream>
#include "train.hpp"
extern tipl::program_option<tipl::out> po;
bool load_from_file(UNet3d& model,const char* file_name);
bool save_to_file(UNet3d& model,const char* file_name);
using namespace std::chrono_literals;

void preprocess_label(UNet3d& model,tipl::image<3>& train_label,const training_param& param)
{
    if(param.relations.empty())
    {
        tipl::expand_label_to_dimension(train_label,model->out_count);
        return;
    }

    tipl::image<3> relation_map(model->dim.multiply(tipl::shape<3>::z,model->out_count));
    for(const auto& each_relation : param.relations)
    {
        std::vector<char> relation_mask(model->out_count+1);
        std::vector<size_t> relation_pose;
        for(auto v : each_relation)
        {
            relation_mask[v+1] = 1;
            relation_pose.push_back(v*model->dim.size());
        }
        float added_weight = 0.5f/float(param.
                                        relations.size());
        for(size_t i = 0;i < train_label.size();++i)
        {
            int label = int(train_label[i]);
            if(!label || !relation_mask[label])
                continue;
            for(auto p : relation_pose)
                relation_map[p+i] += added_weight;
        }
    }

    tipl::expand_label_to_dimension(train_label,model->out_count);

    for(size_t i = 0;i < train_label.size();++i)
        train_label[i] = std::max<float>(relation_map[i],train_label[i]);
}

bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          size_t in_count,
                          tipl::image<3>& input,
                          tipl::image<3>& label,
                          tipl::shape<3>& image_shape)
{
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    {
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(image_name))
            return false;
        nii.get_image_dimension(image_shape);
        if(nii.dim(4) != in_count)
            return false;
        input.resize(tipl::shape<3>(image_shape.width(),image_shape.height(),image_shape.depth()*in_count));
        for(int c = 0;c < in_count;++c)
        {
            auto I = input.alias(image_shape.size()*c,image_shape);
            nii >> I;
        }
        nii.get_image_transformation(image_t);
    }


    tipl::io::gz_nifti nii;
    tipl::shape<3> label_shape;
    if(!nii.load_from_file(label_name))
        return false;
    nii.get_image_dimension(label_shape);
    if(nii.dim(4) > 1)
    {
        label.resize(label_shape);
        for(size_t index = 1;index <= nii.dim(4);++index)
        {
            tipl::image<3> I;
            nii >> I;
            for(size_t pos = 0;pos < I.size();++pos)
                if(I[pos])
                    label[pos] = index;
        }
    }
    else
    {
        nii >> label;
    }
    tipl::matrix<4,4,float> label_t((tipl::identity_matrix()));
    nii.get_image_transformation(label_t);
    if(image_shape != label_shape || label_t != image_t)
    {
        tipl::image<3> new_label(image_shape);
        tipl::resample<tipl::nearest>(label,new_label,tipl::from_space(image_t).to(label_t));
        label.swap(new_label);
    }
    return true;
}


std::vector<size_t> get_label_count(const tipl::image<3>& label,size_t out_count)
{
    std::vector<size_t> sum(out_count);
    for(size_t i = 0;i < label.size();++i)
        if(label[i])
        {
            auto v = label[i]-1;
            if(v < out_count)
                ++sum[v];
        }
    return sum;
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
    test_out_mask.clear();

    // read training data
    read_images.reset(new std::thread([=]()
    {
        // checking template or subject images
        {
            for(size_t i = 0;i < param.image_file_name.size();++i)
            {
                reading_status = "checking ";
                reading_status += param.image_file_name[i];
                tipl::io::gz_nifti in;
                if(!in.load_from_file(param.image_file_name[i].c_str()))
                {
                    error_msg = "invalid NIFTI file: ";
                    error_msg += param.image_file_name[i];
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
                    preprocess_label(model,input_label,param);
                else
                    tipl::normalize(input_label);

                tipl::image<3,char> mask(model->dim.multiply(tipl::shape<3>::z,model->out_count));
                {
                    auto I = mask.alias(0,model->dim);
                    tipl::threshold(input_label,I,0);
                    for(size_t i = 1;i < model->out_count;++i)
                        std::copy(mask.begin(),mask.begin()+model->dim.size(),mask.begin()+model->dim.size()*i);
                }


                try{
                test_in_tensor.push_back(torch::from_blob(input_image.data(),
                    {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_tensor.push_back(torch::from_blob(input_label.data(),
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_mask.push_back(torch::from_blob(mask.data(),
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])},torch::kUInt8).to(param.device));
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
                    preprocess_label(model,out_data_thread,param);

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

void train_unet::update_epoch_count()
{
    error.resize(param.epoch);
    for(auto& each : test_error_foreground)
        each.resize(param.epoch);
    for(auto& each : test_error_background)
        each.resize(param.epoch);

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
    auto run_training = [=](){
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false; }
        } guard(running);

        try{

            std::shared_ptr<torch::optim::Optimizer> optimizer(new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(param.learning_rate)));

            while(!test_data_ready || pause)
            {
                std::this_thread::sleep_for(100ms);
                if(aborted)
                    return;
            }

            tipl::out()     << "1                        0.1                        0.01                   0.001";
            tipl::out()     << "|-------------------------|--------------------------|-------------------------|";
            std::string line = "|                         |                          |                         |";
            auto to_chart = [](float error)->int{return int(std::max<float>(0.0f,std::min<float>(79.0f,(-std::log10(error))*80.0f/3.0f)));};
            for (size_t cur_data_index = 0; cur_epoch < param.epoch && !aborted;cur_epoch++,cur_data_index += param.batch_size)
            {
                if(param.learning_rate != optimizer->defaults().get_lr())
                {
                    tipl::out() << "set learning rate to " << param.learning_rate << std::endl;
                    optimizer->defaults().set_lr(param.learning_rate);
                }

                if(need_output_model || !test_in_tensor.empty())
                {
                    output_model->copy_from(*model);
                    need_output_model = false;
                }

                if(!test_in_tensor.empty())
                {
                    output_model->eval();
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                    {
                        float mse_foreground = 0.0f,mse_background = 0.0f;
                        auto f = output_model->forward(test_in_tensor[i]);
                        mse_foreground = torch::mse_loss(f.masked_select(test_out_mask[i].gt(0)),test_out_tensor[i].masked_select(test_out_mask[i].gt(0))).item().toFloat();
                        mse_background = torch::mse_loss(f.masked_select(test_out_mask[i].le(0)),test_out_tensor[i].masked_select(test_out_mask[i].le(0))).item().toFloat();
                        test_error_foreground[i][cur_epoch] = mse_foreground;
                        test_error_background[i][cur_epoch] = mse_background;
                    }
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
                                training_status += std::to_string(in_data_read_id[thread]);
                                training_status += ".";
                                training_status += std::to_string(model_id);
                                training_status += train_image_is_template[in_data_read_id[thread]] ? "t":"s";
                            }

                            while(!data_ready[thread] || pause)
                            {
                                std::this_thread::sleep_for(100ms);
                                if(aborted)
                                    return;
                            }

                            torch::Tensor out_tensor_thread,in_tensor_thread;
                            out_tensor_thread = torch::from_blob(out_data[thread].data(),
                                    {1,cur_model->out_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(cur_model->device());
                            in_tensor_thread = torch::from_blob(in_data[thread].data(),
                                    {1,cur_model->in_count,int(cur_model->dim[2]),int(cur_model->dim[1]),int(cur_model->dim[0])}).to(cur_model->device());
                            data_ready[thread] = false;

                            auto output = cur_model->forward(in_tensor_thread);
                            auto weight = train_image_is_template[in_data_read_id[thread]] ? param.template_label_weight : param.subject_label_weight;
                            if(weight.size() == cur_model->out_count)
                            {
                                at::Tensor loss;
                                for(size_t i = 0;i < cur_model->out_count;++i)
                                    if(weight[i] != 0.0f)
                                    {
                                        auto l = torch::mse_loss(output.select(1,i),out_tensor_thread.select(1,i))*weight[i];
                                        if(loss.defined())
                                            loss += l;
                                        else
                                            loss = l;
                                    }
                                output = loss;
                            }
                            else
                                output = torch::mse_loss(output,out_tensor_thread);

                            error_each_model[model_id] += output.item().toFloat();
                            output.backward();
                        }
                    }
                    catch(const c10::Error& error)
                    {
                        error_msg = std::string("during ") + training_status + ":" + error.what();
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

                {
                    float sum_error = tipl::sum(error_each_model);
                    float previous_error = cur_epoch > 0 ? error[cur_epoch-1]:sum_error/float(param.batch_size);
                    error[cur_epoch] = previous_error*0.95 + sum_error/float(param.batch_size)*0.05f;
                    auto out = line;
                    out[to_chart(error[cur_epoch])] = 'T';
                    //out[to_chart(mse_foreground)] = 'F';
                    //out[to_chart(mse_background)] = 'B';

                    std::string out2("|epoch:");
                    out2 += std::to_string(cur_epoch);
                    out2 += " error:";
                    out2 += std::to_string(error[cur_epoch]);
                    std::copy(out2.begin(),out2.end(),out.begin());

                    tipl::out() << out;
                }

                if(po.has("network") &&
                   (((cur_epoch + 1) % 500 == 0) || cur_epoch+1 == param.epoch))
                {
                    if(!save_to_file(model,po.get("network").c_str()))
                    {
                        error_msg = "ERROR: failed to save network";
                        aborted = true;
                    }
                    if(!save_error_to(po.get("error",po.get("network") + ".error.txt").c_str()))
                        error_msg = "ERROR: failed to save error";
                }
            }
            tipl::out() << (training_status = "training completed");

        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("during ") + training_status + ":" + error.what();
        }
        catch(...)
        {
            error_msg = "unknown error in training";
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
        error.clear();
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
        tipl::io::gz_nifti in;
        if(!in.load_from_file(param.image_file_name[0]))
        {
            error_msg = "Invalid NIFTI format";
            error_msg += param.image_file_name[0];
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

    test_error_background = test_error_foreground = std::vector<std::vector<float> >(param.test_image_file_name.size());
    update_epoch_count();

    model->to(param.device);
    model->train();
    output_model = UNet3d(model->in_count,model->out_count,model->feature_string);
    output_model->to(param.device);

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
UNet3d& train_unet::get_model(void)
{
    if(running)
    {
        need_output_model = true;
        tipl::progress p("copying network");
        while(need_output_model && p(0,1))
        {
            std::this_thread::sleep_for(100ms);
        }
        if(p.aborted())
            return model;
        return output_model;
    }
    return model;
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
    if(!out)
        return false;
    out << "trainning_error\t";
    for(size_t j = 0;j < test_error_foreground.size();++j)
        out << "test_foreground_error\ttest_background_error\t";
    out << std::endl;
    for(size_t i = 0;i < error.size() && i < cur_epoch;++i)
    {
        out << error[i] << "\t";
        for(size_t j = 0;j < test_error_foreground.size();++j)
        {
            out << test_error_foreground[j][i] << "\t";
            out << test_error_background[j][i] << "\t";
        }
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
        if(!po.get_files("image",train.param.image_file_name) ||
           !po.get_files("label",train.param.label_file_name))
        {
            tipl::out() << "ERROR: " << po.error_msg;
            return 1;
        }

        if(train.param.image_file_name.size() != train.param.label_file_name.size())
        {
            tipl::out() << "ERROR: different number of files found for image and label";
            return 1;
        }
        if(train.param.image_file_name.empty())
        {
            tipl::out() << "ERROR: no available training images";
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
            tipl::out() << "ERROR: failed to load model from " << network;
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
            tipl::out() << "ERROR: cannot open the label file" << train.param.label_file_name[0];
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
            tipl::out() << "ERROR: invalid network structure ";
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
            tipl::out() << "ERROR: cannot load options";
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
        tipl::out() << "ERROR: " << train.error_msg;
        return 1;
    }
    return 0;
}
