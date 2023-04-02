#include "train.hpp"

size_t get_label_out_count(const std::string& label_name)
{
    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(label_name))
        return 0;
    if(nii.dim(4) != 1)
        return nii.dim(4);
    else
    {
        tipl::image<3,float> labels;
        nii >> labels;
        if(tipl::is_label_image(labels))
            return tipl::max_value(labels);
    }
    return 1;
}
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          size_t out_count,
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
        if(nii.dim(4) != out_count)
            return false;
        label.resize(label_shape.multiply(tipl::shape<3>::z,out_count));
        auto label_alias = label.alias(0,tipl::shape<4>(label_shape[0],label_shape[1],label_shape[2],out_count));
        nii >> label_alias;
    }
    else
    {
        nii >> label;
        if(out_count != 1)
            tipl::expand_label_to_dimension(label,out_count);
    }
    tipl::matrix<4,4,float> label_t((tipl::identity_matrix()));
    nii.get_image_transformation(label_t);
    if(image.shape() != label_shape || label_t != image_t)
    {
        tipl::out() << "spatial transform label file to image file space" << std::endl;
        tipl::image<3> new_label(image.shape().multiply(tipl::shape<3>::z,out_count));
        for(size_t i = 0;i < out_count;++i)
        {
            auto I = label.alias(label_shape.size()*i,label_shape);
            auto J = new_label.alias(image.shape().size()*i,image.shape());
            tipl::resample_mt(I,J,tipl::from_space(image_t).to(label_t));
        }
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
void create_dropout_at(image_type& image,image_type& label,const tipl::vector<3,int>& center,const tipl::vector<3,int>& radius)
{
    auto pos = center-radius;
    auto sizes = radius+radius;
    tipl::draw_rect(image,pos,sizes,0);
    if(!label.empty())
    {
        auto out_count = label.depth()/image.depth();
        for(size_t i = 0;i < out_count;++i)
        {
            auto I = label.alias(image.size()*i,image.shape());
            tipl::draw_rect(I,pos,sizes,0);
        }
    }
}

void load_image_and_label(tipl::image<3>& image,
                          tipl::image<3>& label,
                          const tipl::vector<3>& image_vs,
                          const tipl::shape<3>& template_shape)
{
    tipl::uniform_dist<float> one(-1.0f,1.0f,time(0));
    auto range = [&one](float from,float to){return one()*(to-from)*0.5f+(to+from)*0.5f;};
    tipl::vector<3> template_vs(1.0f,1.0f,1.0f);
    if(image_vs[0] < 1.0f)
        template_vs[2] = template_vs[1] = template_vs[0] = image.width()*image_vs[0]/template_shape[0];

    tipl::affine_transform<float> transform = {one()*30.0f*template_vs[0],one()*30.0f*template_vs[0],one()*30.0f*template_vs[0],
                                        one()*0.45f,one()*0.45f/4.0f,one()*0.45f/4.0f,
                                        range(0.75f,2.0f),range(0.75f,2.0f),range(0.75f,2.0f),
                                        one()*0.15f,one()*0.15f,one()*0.15f};



    tipl::image<3,tipl::vector<3> > displaced(template_shape);

    tipl::par_for(int(range(0.0f,4.0f)),[&](int)
    {
        create_distortion_at(displaced,
                             tipl::vector<3,int>(
                                 (image.shape()[0]-1)*range(0.4f,0.6f),
                                 (image.shape()[1]-1)*range(0.4f,0.6f),
                                 (image.shape()[2]-1)*range(0.4f,0.6f)),
                                 (image.shape()[0]-1)*range(0.2f,0.5f), // radius
                                 range(0.05f,0.2f));                    //magnitude
    });

    tipl::par_for(int(range(0.0f,4.0f)),[&](int)
    {
        tipl::vector<3,int> center,radius;
        for(int i = 0;i < 3;++i)
        {
            center[i] = (image.shape()[i]-1)*range(0.2f,0.8f);
            radius[i] = (image.shape()[i]-1)*range(0.05f,0.1f);
        }
        create_dropout_at(image,label,center,radius);
    });


    if(!label.empty())
    {
        auto out_count = label.depth()/image.depth();
        tipl::image<3> label_out(template_shape.multiply(tipl::shape<3>::z,out_count));
        for(size_t i = 0;i < out_count;++i)
        {
            auto J = label_out.alias(template_shape.size()*i,template_shape);
            tipl::compose_displacement_with_affine(label.alias(image.size()*i,image.shape()),J,tipl::transformation_matrix<float>(transform,template_shape,template_vs,image.shape(),image_vs),displaced);
        }
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
            tipl::affine_transform<float> arg= {one()*30.0f,one()*30.0f,one()*30.0f,
                                                one()*2.0f,one()*2.0f,one()*2.0f,
                                                range(0.25f,0.5f),range(0.25f,0.5f),range(0.25f,0.5f),
                                                0.0f,0.0f,0.0f};
            tipl::resample_mt(image,background,tipl::transformation_matrix<float>(arg,template_shape,template_vs,image.shape(),image_vs));
        }
        tipl::normalize(background,0.1f);
        for(size_t i = 0;i < image_out.size();++i)
            image_out[i] += background[i]/(0.1f+image_out[i]);
    }


    image_out.swap(image);
}

void train_unet::read_file(const TrainParam& param)
{
    in_data = std::vector<tipl::image<3> >(param.epoch*param.batch_size);
    out_data = std::vector<tipl::image<3> >(in_data.size());
    data_ready = std::vector<bool>(in_data.size(),false);

    read_file_thread.reset(new std::thread([=]()
    {
        std::vector<tipl::image<3> > image;
        std::vector<tipl::image<3> > label;
        std::vector<tipl::vector<3> > image_vs;

        for(int read_id = 0;read_id < param.image_file_name.size();++read_id)
        {
            tipl::image<3> new_image,new_label;
            tipl::vector<3> new_vs;
            if(!read_image_and_label(param.image_file_name[read_id],
                                     param.label_file_name[read_id],
                                     new_image,new_label,model->out_count,new_vs))
                continue;
            image.push_back(std::move(new_image));
            label.push_back(std::move(new_label));
            image_vs.push_back(new_vs);
        }

        if(image.empty())
        {
            error_msg = "no training image";
            aborted = true;
            return;
        }
        const int thread_count = std::thread::hardware_concurrency()/2;
        tipl::par_for(thread_count,[&](size_t thread)
        {
            for(size_t i = thread;i < in_data.size() && !aborted;)
            {
                while(i > (cur_epoch+2)*param.batch_size || pause)
                {
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                }
                auto read_id = std::rand() % image.size();
                if(!image[read_id].size())
                    continue;
                in_data[i] = image[read_id];
                out_data[i] = label[read_id];
                load_image_and_label(in_data[i],out_data[i],image_vs[read_id],param.dim);
                data_ready[i] = true;
                i += thread_count;
            }
        });
    }));
}

void train_unet::prepare_tensor(const TrainParam& param)
{
    in_tensor = std::vector<torch::Tensor>(param.epoch);
    out_tensor = std::vector<torch::Tensor>(param.epoch);
    tensor_ready = std::vector<bool>(param.epoch,false);
    freed_tensor = 0;
    prepare_tensor_thread.reset(new std::thread([=](){
        try{
            tipl::shape<3> in_dim(param.dim.multiply(tipl::shape<3>::z,param.batch_size*model->in_count)),
                           out_dim(param.dim.multiply(tipl::shape<3>::z,param.batch_size*model->out_count));
            for (size_t i = 0; i < param.epoch && !aborted; i++)
            {
                for(;freed_tensor < cur_epoch;++freed_tensor)
                {
                    in_tensor[freed_tensor] = torch::Tensor();
                    out_tensor[freed_tensor] = torch::Tensor();
                }
                tipl::image<3> in(in_dim),out(out_dim);
                tipl::par_for(param.batch_size,[&](size_t b)
                {
                    size_t j = i*param.batch_size + b;
                    while(!data_ready[j] || pause)
                    {
                        if(aborted)
                            return;
                        using namespace std::chrono_literals;
                        std::this_thread::sleep_for(200ms);

                    }
                    std::copy(in_data[j].begin(),in_data[j].end(),in.begin()+b*in_data[j].size());
                    std::copy(out_data[j].begin(),out_data[j].end(),out.begin()+b*out_data[j].size());
                    tipl::image<3>().swap(in_data[j]);
                    tipl::image<3>().swap(out_data[j]);
                });
                if(aborted)
                    return;
                in_tensor[i] = torch::from_blob(&in[0],
                    {param.batch_size,model->in_count,int(param.dim[2]),int(param.dim[1]),int(param.dim[0])}).to(param.device);
                out_tensor[i] = torch::from_blob(&out[0],
                    {param.batch_size,model->out_count,int(param.dim[2]),int(param.dim[1]),int(param.dim[0])}).to(param.device);
                tensor_ready[i] = true;
            }
        }
        catch(...)
        {
            error_msg = "error occured in preparing tensor";
            std::cout << error_msg << std::endl;
            pause = aborted = true;
        }
    }));
}

void train_unet::train(const TrainParam& param)
{
    error = std::vector<float>(param.epoch);
    optimizer.reset(new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(param.learning_rate)));
    train_thread.reset(new std::thread([=](){
        try{
            for (cur_epoch = 0; cur_epoch < param.epoch && !aborted; cur_epoch++)
            {
                while(!tensor_ready[cur_epoch] || pause)
                {
                    if(aborted)
                    {
                        running = false;
                        return;
                    }
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                }
                optimizer->zero_grad();
                torch::Tensor loss = torch::mse_loss(model->forward(in_tensor[cur_epoch]),out_tensor[cur_epoch]);
                loss.backward();
                optimizer->step();
                std::cout << "epoch:" << cur_epoch << " error:" << (error[cur_epoch] = loss.item().toFloat()) << std::endl;;
            }
        }
        catch(...)
        {
            error_msg = "error occured during training";
            std::cout << error_msg << std::endl;
            pause = aborted = true;
        }
        running = false;
    }));
}
void train_unet::start(const TrainParam& param)
{
    stop();
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

