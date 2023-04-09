#include "train.hpp"

bool get_label_info(const std::string& label_name,int& out_count,bool& is_label)
{
    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(label_name))
        return false;
    if(nii.dim(4) != 1)
        out_count = nii.dim(4);
    if(nii.is_integer())
    {
        is_label = true;
        if(nii.dim(4) == 1)
        {
            tipl::image<3,short> labels;
            nii >> labels;
            out_count = tipl::max_value(labels);
        }
    }
    else
    {
        tipl::image<3,float> labels;
        nii >> labels;
        is_label = tipl::is_label_image(labels);
        if(nii.dim(4) == 1)
            out_count = (is_label ? tipl::max_value(labels) : 1);
    }

    if(out_count > 255)
    {
        out_count = 1;
        is_label = false;
    }
    return true;
}
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          tipl::vector<3>& vs)
{
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    if(!tipl::io::gz_nifti::load_from_file(image_name.c_str(),image,vs,image_t))
        return false;

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
    if(image.shape() != label_shape || label_t != image_t)
    {
        tipl::out() << "spatial transform label file to image file space" << std::endl;
        tipl::image<3> new_label(image.shape());
        tipl::resample_mt<tipl::nearest>(label,new_label,tipl::from_space(image_t).to(label_t));
        label.swap(new_label);
    }
    return true;
}

template<typename image_type>
void intensity_wave(image_type& image_out,tipl::uniform_dist<float>& one,float frequency,float magnitude)
{
    const double pi = std::acos(-1);
    tipl::vector<3> f(one()*frequency*pi,one()*frequency*pi,one()*frequency*pi);
    tipl::vector<3> shift(one()*pi,one()*pi,one()*pi);
    float a = std::abs(one()*magnitude); // [0 0.25]
    float b = 1.0f-a-a;
    for(tipl::pixel_index<3> index(image_out.shape());index < image_out.size();++index)
    {
        tipl::vector<3> pos(index.begin());
        tipl::divide(pos,image_out.shape());
        pos += shift;
        image_out[index.index()] *= ((std::cos(pos*f)+1.0f)*a+b);
    }
}
template<typename image_type>
void ghost(image_type& image_out,int shift,float magnitude,bool direction)
{
    tipl::image<3> ghost = image_out;
    ghost *= magnitude;
    if(direction)
    {
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(shift,0,0));
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(-shift,0,0));
    }
    else
    {
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(0,shift,0));
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(0,-shift,0));
    }
}
template<typename image_type>
void create_distortion_at(image_type& displaced,const tipl::vector<3,int>& center,float radius,float magnitude)
{
    auto radius_5 = radius*magnitude;
    auto pi_2_radius = std::acos(-1)/radius;
    tipl::for_each_neighbors(tipl::pixel_index<3>(center.begin(),displaced.shape()),displaced.shape(),radius,[&](const auto& pos)
    {
        tipl::vector<3> dir(pos);
        dir -= center;
        auto length = dir.length();
        if(length > radius)
            return;
        dir *= -radius_5*std::sin(length*pi_2_radius)/length;
        displaced[pos.index()] += dir;
    });
}

template<typename image_type>
void create_dropout_at(image_type& image,const tipl::vector<3,int>& center,const tipl::vector<3,int>& radius)
{
    if(image.empty())
        return;
    auto pos = center-radius;
    auto sizes = radius+radius;
    tipl::draw_rect(image,pos,sizes,0);
}

void load_image_and_label(tipl::image<3>& image,
                          tipl::image<3>& label,
                          const tipl::vector<3>& image_vs,
                          const tipl::shape<3>& template_shape,
                          size_t random_seed)
{
    tipl::uniform_dist<float> one(-1.0f,1.0f,random_seed);
    auto range = [&one](float from,float to){return one()*(to-from)*0.5f+(to+from)*0.5f;};
    auto random_location = [&range](const tipl::shape<3>& sp,float from,float to)
                    {return tipl::vector<3,int>((sp[0]-1)*range(from,to),(sp[1]-1)*range(from,to),(sp[2]-1)*range(from,to));};

    tipl::vector<3> template_vs(1.0f,1.0f,1.0f);
    if(image_vs[0] < 1.0f)
        template_vs[2] = template_vs[1] = template_vs[0] = image.width()*image_vs[0]/template_shape[0];

    auto resolution = range(0.75f,1.5f);
    tipl::affine_transform<float> transform = {one()*30.0f*template_vs[0],one()*30.0f*template_vs[0],one()*30.0f*template_vs[0],
                                        one()*0.45f,one()*0.45f/4.0f,one()*0.45f/4.0f,
                                        resolution*range(0.8f,1.25f),resolution*range(0.8f,1.25f),resolution*range(0.8f,1.25f),
                                        one()*0.15f,one()*0.15f,one()*0.15f};

    tipl::image<3,tipl::vector<3> > displaced(template_shape);

    tipl::par_for(int(range(0.0f,4.0f)),[&](int)
    {
        create_distortion_at(displaced,random_location(image.shape(),0.4f,0.6f),
                                 (image.shape()[0]-1)*range(0.2f,0.5f), // radius
                                 range(0.05f,0.2f));                    //magnitude
    });


    tipl::par_for(int(range(0.0f,4.0f)),[&](int)
    {
        auto center = random_location(image.shape(),0.2f,0.8f);
        auto radius = random_location(image.shape(),0.05f,0.1f);
        create_dropout_at(image,center,radius);
        create_dropout_at(label,center,radius);
    });



    if(!label.empty())
    {
        tipl::image<3> label_out(template_shape);
        tipl::compose_displacement_with_affine<tipl::nearest>(label,label_out,tipl::transformation_matrix<float>(transform,template_shape,template_vs,image.shape(),image_vs),displaced);
        label_out.swap(label);
    }


    tipl::image<3> image_out(template_shape);

    {
        tipl::image<3> reduced_image(image.shape());
        tipl::vector<3> dd(range(0.5f,1.0f),range(0.5f,1.0f),range(0.5f,1.0f));
        tipl::vector<3> new_vs(image_vs[0]/dd[0],image_vs[1]/dd[1],image_vs[2]/dd[2]);
        tipl::affine_transform<float> image_arg;
        image_arg.scaling[0] = 1.0f/dd[0];
        image_arg.scaling[1] = 1.0f/dd[1];
        image_arg.scaling[2] = 1.0f/dd[2];
        tipl::resample_mt(image,reduced_image,
            tipl::transformation_matrix<float>(image_arg,image.shape(),new_vs,image.shape(),image_vs));
        image_arg = transform;
        image_arg.scaling[0] *= dd[0];
        image_arg.scaling[1] *= dd[1];
        image_arg.scaling[2] *= dd[2];
        tipl::compose_displacement_with_affine(reduced_image,image_out,
            tipl::transformation_matrix<float>(image_arg,template_shape,template_vs,image.shape(),new_vs),displaced);
    }


    if(one() > 0)
        ghost(image_out,range(0.05f,0.25f)*image_out.width(),0.125f,one() > 0);

    intensity_wave(image_out,one,2.0f,0.25f); // low frequency at 0.25 magnitude
    intensity_wave(image_out,one,20.0f,0.1f); // high frequency at 0.1 magnitude


    tipl::normalize(image_out);
    tipl::lower_threshold(image_out,0.0f);

    if(one() > 0)
    {
        tipl::image<3> background(template_shape);
        {
            auto resolution = range(0.05f,0.1f);
            tipl::affine_transform<float> arg= {one()*30.0f,one()*30.0f,one()*30.0f,
                                                one()*2.0f,one()*2.0f,one()*2.0f,
                                                resolution,resolution,resolution,
                                                0.0f,0.0f,0.0f};
            tipl::resample_mt(image,background,tipl::transformation_matrix<float>(arg,template_shape,template_vs,image.shape(),image_vs));
        }
        tipl::normalize(background,0.05f);
        for(size_t i = 0;i < image_out.size();++i)
            image_out[i] += background[i]/(0.1f+image_out[i]);
    }


    image_out.swap(image);

    if(one() > 0.0f)
    {
        auto center = random_location(image.shape(),0.2f,0.8f);
        auto size = random_location(image.shape(),0.0f,0.1f);
        create_dropout_at(image,center,size);
        create_dropout_at(label,center,size);
    }
}

auto get_weight(const tipl::image<3>& label,size_t out_count)
{
    std::vector<size_t> sum(out_count);
    for(size_t i = 0;i < label.size();++i)
        if(label[i])
        {
            auto v = label[i]-1;
            if(v < out_count)
                ++sum[v];
        }
    auto max_value = tipl::max_value(sum);
    std::vector<float> weight(out_count);
    for(size_t i = 0;i < out_count;++i)
        weight[i] = (sum[i] ? float(max_value)/float(sum[i]) : 0.0f);
    return weight;
}
tipl::shape<3> unet_inputsize(const tipl::shape<3>& s);

void train_unet::read_file(const TrainParam& param)
{
    in_data = std::vector<tipl::image<3> >(param.epoch*param.batch_size);
    out_data = std::vector<tipl::image<3> >(in_data.size());
    out_data_weight = std::vector<std::vector<float> >(in_data.size());
    data_ready = std::vector<bool>(in_data.size(),false);

    test_in_tensor.clear();
    test_out_tensor.clear();
    test_error.clear();

    read_file_thread.reset(new std::thread([=]()
    {
        std::vector<tipl::image<3> > image;
        std::vector<tipl::image<3> > label;
        std::vector<std::vector<float> > label_weight;
        std::vector<tipl::vector<3> > image_vs;

        // prepare training data
        {
            status = "reading input data";
            tipl::progress prog("read training data");
            for(int read_id = 0;read_id < param.image_file_name.size();++read_id)
            {
                tipl::out() << "reading " << param.image_file_name[read_id] << std::endl;
                tipl::out() << "reading " << param.label_file_name[read_id] << std::endl;
                tipl::image<3> new_image,new_label;
                tipl::vector<3> new_vs;

                if(!read_image_and_label(param.image_file_name[read_id],
                                         param.label_file_name[read_id],
                                         new_image,new_label,new_vs))
                    continue;
                label_weight.push_back(model->out_count > 1 ? get_weight(new_label,model->out_count) : std::vector<float>(model->out_count));
                image.push_back(std::move(new_image));
                label.push_back(std::move(new_label));
                image_vs.push_back(new_vs);
                if(model->out_count > 1)
                {
                    std::ostringstream out;
                    std::copy(label_weight.back().begin(),label_weight.back().end(),std::ostream_iterator<float>(out," "));
                    tipl::out() << "label weightes: " << out.str();
                }
            }
        }

        //prepare test data
        {
            tipl::progress prog("read test data");
            for(int read_id = 0;read_id < param.test_image_file_name.size();++read_id)
            {
                tipl::out() << "reading test image " << param.test_image_file_name[read_id] << std::endl;
                tipl::out() << "reading test label " << param.test_label_file_name[read_id] << std::endl;
                tipl::image<3> new_image,new_label;
                tipl::vector<3> new_vs;
                if(!read_image_and_label(param.test_image_file_name[read_id],
                                         param.test_label_file_name[read_id],
                                         new_image,new_label,new_vs))
                    continue;
                auto new_shape = unet_inputsize(new_image.shape());
                if(new_shape != new_image.shape())
                {
                    tipl::image<3> image(new_shape),label(new_shape);
                    tipl::draw(new_image,image,tipl::vector<3,int>(0,0,0));
                    tipl::draw(new_label,label,tipl::vector<3,int>(0,0,0));
                    new_image.swap(image);
                    new_label.swap(label);
                }
                tipl::expand_label_to_dimension(new_label,model->out_count);
                test_in_tensor.push_back(torch::from_blob(&new_image[0],
                    {1,model->in_count,int(new_image.shape()[2]),int(new_image.shape()[1]),int(new_image.shape()[0])}).to(param.device));
                test_out_tensor.push_back(torch::from_blob(&new_label[0],
                    {1,model->out_count,int(new_image.shape()[2]),int(new_image.shape()[1]),int(new_image.shape()[0])}).to(param.device));
            }
            if(!test_in_tensor.empty())
                test_error.resize(param.epoch);
        }
        if(image.empty())
        {
            error_msg = "no training image";
            aborted = true;
            return;
        }
        const int thread_count = std::thread::hardware_concurrency();
        tipl::par_for(thread_count,[&](size_t thread)
        {
            tipl::uniform_dist<float> retaining(-1.0f,1.0f,thread);
            for(size_t i = thread;i < in_data.size() && !aborted;)
            {
                while(i > cur_data_index+8 || pause)
                {
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    status = "network training";
                    std::this_thread::sleep_for(100ms);
                }
                auto read_id = std::rand() % image.size();
                if(!image[read_id].size())
                    continue;
                in_data[i] = image[read_id];
                out_data[i] = label[read_id];
                out_data_weight[i] = label_weight[read_id];
                load_image_and_label(in_data[i],out_data[i],image_vs[read_id],param.dim,i);
                tipl::expand_label_to_dimension(out_data[i],model->out_count);
                data_ready[i] = true;
                i += thread_count;
            }
        });
    }));
}

void train_unet::prepare_tensor(const TrainParam& param)
{
    in_tensor = std::vector<torch::Tensor>(data_ready.size());
    out_tensor = std::vector<torch::Tensor>(data_ready.size());
    out_tensor_weight = std::vector<torch::Tensor>(data_ready.size());
    tensor_ready = std::vector<bool>(data_ready.size(),false);
    prepare_tensor_thread.reset(new std::thread([=](){
        try{
            const int thread_count = 2;
            tipl::par_for(thread_count,[&](size_t thread)
            {
                for (size_t i = thread; i < data_ready.size() && !aborted; i += thread_count)
                {
                    while(i >= cur_data_index+4 || !data_ready[i] || pause)
                    {
                        if(aborted)
                            return;
                        using namespace std::chrono_literals;
                        status = "image deformation and transformation";
                        std::this_thread::sleep_for(100ms);
                    }
                    if(aborted)
                        return;
                    in_tensor[i] = torch::from_blob(&in_data[i][0],
                        {1,model->in_count,int(param.dim[2]),int(param.dim[1]),int(param.dim[0])}).to(param.device);
                    out_tensor[i] = torch::from_blob(&out_data[i][0],
                        {1,model->out_count,int(param.dim[2]),int(param.dim[1]),int(param.dim[0])}).to(param.device);
                    out_tensor_weight[i] = torch::from_blob(&out_data_weight[i][0],
                        {model->out_count}).to(param.device);
                    tensor_ready[i] = true;
                    tipl::image<3>().swap(in_data[i]);
                    tipl::image<3>().swap(out_data[i]);
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

void train_unet::train(const TrainParam& param)
{
    error = std::vector<float>(param.epoch);
    optimizer.reset(new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(param.learning_rate)));
    cur_data_index = 0;
    cur_epoch = 0;
    train_thread.reset(new std::thread([=](){
        try{
            for (; cur_epoch < param.epoch && !aborted; cur_epoch++)
            {
                float cur_error = 0.0f;
                size_t cur_error_count = 0;
                for(size_t b = 0;b < param.batch_size && !aborted;++b,++cur_data_index)
                {
                    while(!tensor_ready[cur_data_index] || pause)
                    {
                        if(aborted)
                        {
                            running = false;
                            return;
                        }
                        status = "tensor allocation";
                        using namespace std::chrono_literals;
                        std::this_thread::sleep_for(100ms);
                    }
                    /*
                    status = "training with weighted MSE";
                    torch::Tensor loss = (torch::mse_loss(model->forward(in_tensor[cur_data_index]),out_tensor[cur_data_index],at::Reduction::None).
                                        mean({-3,-2,-1})*out_tensor_weight[cur_data_index]).mean();
                    */
                    status = "training";
                    torch::Tensor loss = torch::mse_loss(model->forward(in_tensor[cur_data_index]),out_tensor[cur_data_index]);
                    status = "backward propagation";
                    cur_error += loss.item().toFloat();
                    ++cur_error_count;
                    loss.backward();
                    in_tensor[cur_data_index].reset();
                    out_tensor[cur_data_index].reset();
                    out_tensor_weight[cur_data_index].reset();

                    if(b+1 == param.batch_size ||
                       (    param.from_scratch &&
                            (b+1)%int(std::pow(2,std::floor(std::log2(cur_epoch/2+1)))) == 0) )
                    {
                        status = "step";
                        optimizer->step();
                        optimizer->zero_grad();
                    }
                }
                tipl::out() << "epoch:" << cur_epoch << " error:" << (error[cur_epoch] = cur_error/float(cur_error_count)) << std::endl;
                if(!test_in_tensor.empty())
                {
                    torch::NoGradGuard no_grad;
                    model->set_requires_grad(false);
                    model->set_bn_tracking_running_stats(false);
                    model->eval();
                    float sum_error = 0.0f;
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                        sum_error += torch::mse_loss(model->forward(test_in_tensor[i]),test_out_tensor[i]).item().toFloat();
                    tipl::out() << "epoch:" << cur_epoch << " test error:" << (test_error[cur_epoch] = sum_error/float(test_in_tensor.size())) << std::endl;
                    model->set_requires_grad(true);
                    model->set_bn_tracking_running_stats(true);
                    model->train();

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
        status = "complete";
        running = false;
    }));
}
void train_unet::start(const TrainParam& param)
{
    stop();
    status = "initializing";
    model->to(param.device);
    model->train();
    pause = aborted = false;
    running = true;
    error_msg.clear();
    read_file(param);
    prepare_tensor(param);
    train(param);
}
void train_unet::stop(void)
{
    pause = aborted = true;
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
    cur_epoch = 0;
}

