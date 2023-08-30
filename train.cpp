#include "train.hpp"

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
                          tipl::shape<3>& image_shape,
                          tipl::vector<3>& vs)
{
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    {
        tipl::io::gz_nifti nii;
        if(!nii.load_from_file(image_name))
            return false;
        nii.get_image_dimension(image_shape);
        nii.get_voxel_size(vs);
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
            tipl::par_for(I.size(),[&](size_t pos)
            {
                if(I[pos])
                    label[pos] = index;
            });
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
        tipl::resample_mt<tipl::nearest>(label,new_label,tipl::from_space(image_t).to(label_t));
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
    int thread_count = std::min<int>(8,std::thread::hardware_concurrency());

    train_image = std::vector<tipl::image<3> >(param.image_file_name.size());
    train_label = std::vector<tipl::image<3> >(param.image_file_name.size());
    train_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    train_image_ready = std::vector<bool>(param.image_file_name.size(),false);
    read_train_images.reset(new std::thread([=]()
    {
        // prepare training data
        tipl::progress prog("read training data");
        tipl::out() << "a total of " << param.image_file_name.size() << " training dataset" << std::endl;
        for(int read_id = 0;read_id < param.image_file_name.size() && !aborted;++read_id)
        {
            status = "reading training data";
            tipl::shape<3> image_shape;
            if(!read_image_and_label(param.image_file_name[read_id],param.label_file_name[read_id],
                                     model->in_count,train_image[read_id],train_label[read_id],image_shape,train_image_vs[read_id]))
            {
                error_msg = "cannot read image or label data for ";
                error_msg += std::filesystem::path(param.image_file_name[read_id]).filename().string();
                aborted = true;
                return;
            }
            preprocessing(train_image[read_id],train_label[read_id],image_shape,model->dim);
            if(!param.is_label)
                tipl::normalize(train_label[read_id]);
            train_image_ready[read_id] = true;
        }
    }));


    in_data = std::vector<tipl::image<3> >(thread_count);
    in_data_read_id = std::vector<size_t>(thread_count);
    out_data = std::vector<tipl::image<3> >(thread_count);
    data_ready = std::vector<bool>(thread_count,false);


    test_data_ready = false;
    test_in_tensor.clear();
    test_out_tensor.clear();
    test_out_mask.clear();

    read_file_thread.reset(new std::thread([=]()
    {
        //prepare test data
        {
            tipl::progress prog("read test data");
            for(int read_id = 0;read_id < param.test_image_file_name.size() && !aborted;++read_id)
            {
                while(pause)
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }

                tipl::image<3> input_image,input_label;
                tipl::shape<3> input_shape;
                tipl::vector<3> input_vs;

                if(!read_image_and_label(param.test_image_file_name[read_id],
                                         param.test_label_file_name[read_id],
                                         model->in_count,
                                         input_image,input_label,input_shape,input_vs))
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


                test_in_tensor.push_back(torch::from_blob(&input_image[0],
                    {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_tensor.push_back(torch::from_blob(&input_label[0],
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_mask.push_back(torch::from_blob(&mask[0],
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])},torch::kUInt8).to(param.device));
            }
            tipl::out() << "a total of " << test_out_tensor.size() << " testing dataset" << std::endl;
            test_data_ready = true;
        }


        std::vector<size_t> template_indices;
        std::vector<size_t> non_template_indices;
        for(size_t i = 0;i < param.image_setting.size();++i)
            if(param.image_setting[i].is_template)
                template_indices.insert(template_indices.end(),i);
            else
                non_template_indices.insert(non_template_indices.end(),i);

        model->voxel_size = train_image_vs[0];
        tipl::par_for(thread_count,[&](size_t thread)
        {
            int seed = thread + model->total_training_count;
            int b = thread;
            int non_template_base = 0;
            while(!aborted)
            {
                while(b >= param.batch_size)
                {
                    b -= param.batch_size;
                    non_template_base += std::max<int>(0,int(param.batch_size)-int(template_indices.size()));
                }
                int read_id = 0;
                if(non_template_indices.empty() || b < template_indices.size())
                    read_id = template_indices[b % template_indices.size()];
                else
                    read_id = non_template_indices[(non_template_base+b-template_indices.size())%non_template_indices.size()];

                while(!train_image_ready[read_id] || data_ready[thread] || pause)
                {
                    using namespace std::chrono_literals;
                    if(data_ready[thread])
                        status = "training network";
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }
                if(!train_image[read_id].size())
                    continue;
                in_data[thread] = train_image[read_id];
                out_data[thread] = train_label[read_id];
                in_data_read_id[thread] = read_id;
                visual_perception_augmentation(options,in_data[thread],out_data[thread],param.is_label,model->dim,train_image_vs[read_id],seed);
                if(model->out_count > 1)
                    preprocess_label(model,out_data[thread],param);
                data_ready[thread] = true;
                seed += thread_count;
                b += thread_count;
            }
        });
    }));
}

void train_unet::prepare_tensor(void)
{
    const int thread_count = std::min<int>(8,std::thread::hardware_concurrency());
    in_tensor = std::vector<torch::Tensor>(thread_count);
    in_tensor_read_id = std::vector<size_t>(thread_count);
    out_tensor = std::vector<torch::Tensor>(thread_count);
    tensor_ready = std::vector<bool>(thread_count,false);
    prepare_tensor_thread.reset(new std::thread([=](){
        try{
            tipl::par_for(thread_count,[&](size_t thread)
            {
                size_t i = thread;
                while(!aborted)
                {
                    size_t data_index = i%data_ready.size();
                    while(!data_ready[data_index] || tensor_ready[thread] || pause)
                    {
                        using namespace std::chrono_literals;
                        status = "image deformation and transformation";
                        std::this_thread::sleep_for(100ms);
                        if(aborted)
                            return;
                    }
                    in_tensor[thread] = torch::from_blob(&in_data[data_index][0],
                        {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device);
                    out_tensor[thread] = torch::from_blob(&out_data[data_index][0],
                        {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device);
                    in_tensor_read_id[thread] = in_data_read_id[data_index];
                    tensor_ready[thread] = true;
                    data_ready[data_index] = false;
                    i += thread_count;
                }
            });
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error in preparing tensor:") + error.what();
            pause = aborted = true;
        }
        catch(...)
        {
            pause = aborted = true;
        }
        tipl::out() << error_msg << std::endl;
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
void train_unet::train(void)
{
    error.clear();
    test_error_foreground = std::vector<std::vector<float> >(param.test_image_file_name.size());
    test_error_background = test_error_foreground;
    update_epoch_count();

    cur_epoch = 0;

    output_model = UNet3d(model->in_count,model->out_count,model->feature_string);
    output_model->to(param.device);
    output_model->copy_from(*model);


    train_thread.reset(new std::thread([=](){
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false; }
        } guard(running);

        try{

            size_t cur_data_index = 0;
            size_t best_epoch = 0;

            optimizer.reset(new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(param.learning_rate)));

            while(!test_data_ready || pause)
            {
                if(aborted)
                    return;
                status = "test tensor allocation";
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(100ms);
            }

            for (; cur_epoch < param.epoch && !aborted;cur_epoch++)
            {
                if(param.learning_rate != optimizer->defaults().get_lr())
                {
                    tipl::out() << "set learning rate to " << param.learning_rate << std::endl;
                    optimizer->defaults().set_lr(param.learning_rate);
                }
                std::ostringstream out;

                if(!test_in_tensor.empty())
                {
                    model->eval();
                    out << "epoch:" << cur_epoch << " testing error:";
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                    {
                        auto f = model->forward(test_in_tensor[i]);
                        float mse_foreground = torch::mse_loss(f.masked_select(test_out_mask[i].gt(0)),test_out_tensor[i].masked_select(test_out_mask[i].gt(0))).item().toFloat();
                        float mse_background = torch::mse_loss(f.masked_select(test_out_mask[i].le(0)),test_out_tensor[i].masked_select(test_out_mask[i].le(0))).item().toFloat();
                        test_error_foreground[i][cur_epoch] = mse_foreground;
                        test_error_background[i][cur_epoch] = mse_background;
                        out << mse_foreground << " " << mse_background << " ";
                    }
                    if(test_error_foreground[0][best_epoch] + test_error_background[0][best_epoch] <=
                       test_error_foreground[0][cur_epoch]  + test_error_background[0][cur_epoch])
                        best_epoch = cur_epoch;
                    if(param.output_model_type == 0 || best_epoch == cur_epoch)
                        output_model->copy_from(*model);
                }

                std::string source_str;
                float sum_error = 0.0f;
                model->train();
                for(size_t b = 0;b < param.batch_size && !aborted;++b,++cur_data_index)
                {
                    size_t data_index = cur_data_index%tensor_ready.size();
                    while(!tensor_ready[data_index] || pause)
                    {
                        if(aborted)
                            return;
                        status = "tensor allocation";
                        using namespace std::chrono_literals;
                        std::this_thread::sleep_for(100ms);
                    }

                    if(b)
                        source_str += ",";
                    source_str += std::to_string(in_tensor_read_id[data_index]);
                    source_str += param.image_setting[in_tensor_read_id[data_index]].is_template ? "t":"s";

                    status = "training";
                    auto output = model->forward(in_tensor[data_index]);
                    if(param.label_weight.size() == model->out_count)
                    {
                        at::Tensor loss;
                        for(size_t i = 0;i < model->out_count;++i)
                            if(param.label_weight[i] != 0.0f)
                            {
                                auto l = torch::mse_loss(output.select(1,i),out_tensor[data_index].select(1,i))*param.label_weight[i];
                                if(loss.defined())
                                    loss += l;
                                else
                                    loss = l;
                            }
                        output = loss;
                    }
                    else
                        output = torch::mse_loss(output,out_tensor[data_index]);

                    sum_error += output.item().toFloat();
                    output.backward();
                    tensor_ready[data_index] = false;
                }

                {
                    optimizer->step();
                    optimizer->zero_grad();
                    model->total_training_count += param.batch_size;

                    float previous_error = cur_epoch > 0 ? error[cur_epoch-1]:sum_error/float(param.batch_size);
                    out << " training error:" << (error[cur_epoch] = previous_error*0.95 + sum_error/float(param.batch_size)*0.05f) << std::endl;
                    tipl::out() << out.str();
                    tipl::out() << source_str;

                }

            }
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error in training:") + error.what();
        }
        catch(...)
        {
        }
        tipl::out() << error_msg << std::endl;
        pause = aborted = true;
        if(param.output_model_type == 0)
            output_model->copy_from(*model); // select the lastest model
        else
            model->copy_from(*output_model); // select the low error model
        status = "complete";
    }));
}
void train_unet::start(void)
{
    stop();
    status = "initializing";
    model->to(param.device);
    model->train();
    pause = aborted = false;
    running = true;
    error_msg.clear();
    read_file();
    prepare_tensor();
    train();
}
void train_unet::stop(void)
{
    pause = aborted = true;
    if(read_train_images.get())
    {
        read_train_images->join();
        read_train_images.reset();
    }
    if(read_file_thread.get())
    {
        read_file_thread->join();
        read_file_thread.reset();
    }
    if(prepare_tensor_thread.get())
    {
        prepare_tensor_thread->join();
        prepare_tensor_thread.reset();
    }
    if(train_thread.get())
    {
        train_thread->join();
        train_thread.reset();
    }
}

