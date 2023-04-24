#include "evaluate.hpp"
#include "optiontablewidget.hpp"

tipl::shape<3> unet_inputsize(const tipl::shape<3>& s)
{
    return tipl::shape<3>(int(std::ceil(float(s[0])/32.0f))*32,int(std::ceil(float(s[1])/32.0f))*32,int(std::ceil(float(s[2])/32.0f))*32);
}

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
bool preproc_actions(tipl::image<3>& image,
                   tipl::vector<3>& image_vs,
                   const tipl::shape<3>& input_dim,
                   const tipl::vector<3>& input_vs,
                   unsigned char preproc_strategy,
                   std::string& error_msg)
{
    if(input_dim == image.shape() && image_vs == input_vs)
    {
        tipl::out() << "image resolution and dimension are the same as training data. No padding or regrinding needed.";
        return true;
    }

    switch(preproc_strategy)
    {
        case 0: //match voxel size
        if(image_vs != input_vs)
        {
            tipl::image<3> new_sized_image(unet_inputsize(tipl::shape<3>(
                    float(image.width()*image_vs[0])/input_vs[0],
                    float(image.height()*image_vs[1])/input_vs[1],
                    float(image.depth()*image_vs[2])/input_vs[2])));
            tipl::out() << "regriding to a resolution of " << input_vs << " with a size of " << new_sized_image.shape();
            tipl::resample_mt(image,new_sized_image,
                    tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                        new_sized_image.shape(),input_vs,image.shape(),image_vs));
            new_sized_image.swap(image);
            return true;
        }
        case 2: //original
        {
            tipl::out() << "image has a different dimension. padding or cropping applied";
            tipl::image<3> new_sized_image(unet_inputsize(image.shape()));
            auto shift = tipl::vector<3,int>(new_sized_image.shape())-
                         tipl::vector<3,int>(image.shape());
            shift[2] = 0;
            tipl::draw(image,new_sized_image,shift);
            new_sized_image.swap(image);
        }
        return true;
        case 1: //match sizes
        {
            tipl::out() << "regriding applied.";
            float target_vs = std::min({float(image.width())*image_vs[0]/float(input_dim[0]),
                                        float(image.height())*image_vs[1]/float(input_dim[1]),
                                        float(image.depth())*image_vs[2]/float(input_dim[2])});
            tipl::vector<3> new_input_vs(target_vs,target_vs,target_vs);
            tipl::image<3> new_sized_image(input_dim);
            tipl::resample_mt(image,new_sized_image,
                    tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                        input_dim,new_input_vs,image.shape(),image_vs));
            new_sized_image.swap(image);
        }
        return true;
    }
    return false;
}

template<typename T,typename U>
void reduce_mt(const T& in,U& out,size_t gap = 0)
{
    if(gap == 0)
        gap = out.size();
    tipl::par_for(out.size(),[&](size_t j)
    {
        auto v = out[j];
        for(size_t pos = j;pos < in.size();pos += gap)
            v += in[pos];
        out[j] = v;
    });
}

void postproc_actions(const std::string& command,
                      float param1,float param2,
                      tipl::image<3>& this_image,
                      const tipl::shape<3>& dim,
                      char& is_label)
{
    auto this_image_frames = this_image.depth()/dim[2];
    if(this_image.empty())
        return;
    tipl::out() << "run " << command;
    if(command == "remove_background")
    {
        float remove_background_threshold = param1;
        float remove_background_smoothing = param2;
        tipl::image<3> sum_image(dim);
        reduce_mt(this_image,sum_image);

        for(size_t i = 0;i < remove_background_smoothing;++i)
            tipl::filter::gaussian(sum_image);

        auto threshold = tipl::max_value(sum_image)*remove_background_threshold;
        tipl::image<3> mask;
        tipl::threshold(sum_image,mask,threshold,1,0);
        tipl::morphology::defragment(mask);

        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            for(size_t pos = 0;pos < dim.size();++pos)
                if(!mask[pos])
                    I[pos] = 0;
        });
        return;
    }
    if(command == "defragment")
    {
        float defragment_threshold = param1;
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::image<3,char> mask(I.shape());
            for(size_t i = 0;i < I.size();++i)
                mask[i] = (I[i] > defragment_threshold ? 1:0);
            tipl::morphology::defragment(mask);
            for(size_t i = 0;i < I.size();++i)
                if(!mask[i])
                    I[i] = 0;
        });
        return;
    }
    if(command == "normalize_each")
    {
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::normalize(I);
        });
        is_label = false;
        return;
    }
    if(command == "gaussian_smoothing")
    {
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::filter::gaussian(I);
        });
        is_label = false;
        return;
    }
    if(command =="anisotropic_smoothing")
    {
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::filter::anisotropic_diffusion(I);
        });
        is_label = false;
        return;
    }

    if(command =="normalize_all")
    {
        tipl::image<3> sum_image(dim);
        reduce_mt(this_image,sum_image);

        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            for(size_t pos = 0;pos < dim.size();++pos)
                if(sum_image[pos] != 0.0f)
                    I[pos] /= sum_image[pos];
        });
        is_label = false;
        return;
    }
    if(command =="soft_max")
    {
        float soft_min_prob = param1;
        float soft_max_prob = param2;
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            float m = 0.0f;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                if(this_image[i] > m)
                    m = this_image[i];
            if(m <= soft_min_prob)
            {
                for(size_t i = pos;i < this_image.size();i += dim.size())
                    this_image[i] = 0.0f;
                return;
            }
            m *= soft_max_prob;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                this_image[i] = (this_image[i] >= m ? 1.0f:0.0f);
        });
        is_label = true;
        return;
    }
    if(command =="convert_to_3d")
    {
        tipl::image<3> I(dim);
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            for(size_t i = pos,label = 1;i < this_image.size();i += dim.size(),++label)
                if(this_image[i])
                {
                    I[pos] = label;
                    return;
                }
        });
        I.swap(this_image);
        is_label = true;
        return;
    }
    tipl::out() << "ERROR: unknown command " << command << std::endl;
}
void evaluate_unet::read_file(void)
{
    network_input = std::vector<tipl::image<3> >(param.image_file_name.size());
    raw_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    raw_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    raw_image_flip_swap = std::vector<std::vector<char> >(param.image_file_name.size());
    data_ready = std::vector<bool> (param.image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        for(size_t i = 0;i < network_input.size() && !aborted;++i)
        {
            while(i > cur_prog+6)
            {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(200ms);
                if(aborted)
                    return;
                status = "evaluating";
            }
            tipl::out() << "reading " << param.image_file_name[i] << std::endl;
            tipl::io::gz_nifti in;
            if(!in.load_from_file(param.image_file_name[i]))
            {
                error_msg = in.error_msg;
                aborted = true;
                return;
            }
            in >> network_input[i];
            in.get_voxel_size(raw_image_vs[i]);
            raw_image_flip_swap[i] = in.flip_swap_seq;
            raw_image_shape[i] = network_input[i].shape();
            tipl::out() << "dim: " << network_input[i].shape() << " vs:" << raw_image_vs[i] << std::endl;
            if(!preproc_actions(network_input[i],
                          raw_image_vs[i],
                          model->dim,model->voxel_size,
                          preproc_strategy,error_msg))
            {
                aborted = true;
                return;
            }
            data_ready[i] = true;
        }
    }));
}

void evaluate_unet::evaluate(void)
{
    network_output  = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_thread.reset(new std::thread([=](){
        try{
            for (cur_prog = 0; cur_prog < network_input.size() && !aborted; cur_prog++)
            {
                auto& cur_input = network_input[cur_prog];
                while(!data_ready[cur_prog])
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    if(aborted)
                        return;
                    status = "preproc_actions";
                }
                if(cur_input.empty())
                    continue;
                auto out = model->forward(torch::from_blob(&cur_input[0],
                                          {1,model->in_count,int(cur_input.depth()),int(cur_input.height()),int(cur_input.width())}).to(param.device));
                network_output[cur_prog].resize(cur_input.shape().multiply(tipl::shape<3>::z,model->out_count));
                std::memcpy(&network_output[cur_prog][0],out.to(torch::kCPU).data_ptr<float>(),network_output[cur_prog].size()*sizeof(float));
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
void evaluate_unet::proc_actions(const char* cmd,float param1,float param2)
{
    postproc_actions(cmd,param1,param2,label_prob[cur_output],
                 raw_image_shape[cur_output],
                 is_label[cur_output]);
}
void evaluate_unet::output(void)
{
    label_prob = std::vector<tipl::image<3> >(param.image_file_name.size());
    is_label = std::vector<char>(param.image_file_name.size());
    output_thread.reset(new std::thread([this]()
    {
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false; }
        } guard(running);

        try{
            for (cur_output = 0;cur_output < label_prob.size() && !aborted; cur_output++)
            {
                while(cur_output >= cur_prog)
                {
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    status = "evaluating";
                }
                if(network_output[cur_output].empty())
                    continue;
                tipl::shape<3> dim_from(network_input[cur_output].shape()),
                               dim_to(raw_image_shape[cur_output]);
                label_prob[cur_output].resize(dim_to.multiply(tipl::shape<3>::z,model->out_count));
                tipl::par_for(model->out_count,[&](int i)
                {
                    auto from = network_output[cur_output].alias(dim_from.size()*i,dim_from);
                    auto to = label_prob[cur_output].alias(dim_to.size()*i,dim_to);
                    switch(preproc_strategy)
                    {
                    case 0: //match resolution
                        if(raw_image_vs[cur_output] != model->voxel_size)
                        {
                            tipl::resample_mt(from,to,
                                tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                                    dim_to,raw_image_vs[cur_output],
                                    dim_from,model->voxel_size));
                            return;
                        }
                    case 2: //original
                        {
                            auto shift = tipl::vector<3,int>(to.shape())-tipl::vector<3,int>(from.shape());
                            shift[2] = 0;
                            tipl::draw(from,to,shift);
                        }
                        return;
                    case 1: //match sizes
                    {
                        float target_vs = std::min({float(raw_image_shape[cur_output].width())*raw_image_vs[cur_output][0]/float(model->dim[0]),
                                                    float(raw_image_shape[cur_output].height())*raw_image_vs[cur_output][1]/float(model->dim[1]),
                                                    float(raw_image_shape[cur_output].depth())*raw_image_vs[cur_output][2]/float(model->dim[2])});
                        tipl::resample_mt(from,to,
                            tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                                dim_to,raw_image_vs[cur_output],
                                dim_from,tipl::vector<3>(target_vs,target_vs,target_vs)));
                        return;
                    }

                    }
                });

                switch(postproc_strategy)
                {
                    case 0: // 3d labels
                        proc_actions("normalize_each");
                        proc_actions("remove_background",param.prob_threshold*0.5f,0);
                        proc_actions("normalize_each");
                        proc_actions("remove_background",param.prob_threshold,0);
                        proc_actions("normalize_each");
                        proc_actions("soft_max",0.5f,1.0f);
                        proc_actions("convert_to_3d");
                    break;
                    case 1: // 4d proc maps
                        proc_actions("normalize_each");
                        proc_actions("remove_background",param.prob_threshold*0.5f,0);
                        proc_actions("normalize_each");
                        proc_actions("remove_background",param.prob_threshold,0);
                        proc_actions("normalize_all");
                    break;
                }
                network_input[cur_output] = tipl::image<3>();
                network_output[cur_output] = tipl::image<3>();
            }
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error during output:") + error.what();
        }
        catch(...)
        {
        }
        tipl::out() << error_msg << std::endl;
        aborted = true;
        status = "complete";
    }));
}


void evaluate_unet::start(void)
{
    status = "initiating";
    stop();
    model->to(param.device);
    model->set_requires_grad(false);
    model->set_bn_tracking_running_stats(false);
    model->eval();
    aborted = false;
    running = true;
    error_msg.clear();
    read_file();
    evaluate();
    output();
}
void evaluate_unet::stop(void)
{
    aborted = true;
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
bool evaluate_unet::save_to_file(size_t currentRow,const char* file_name)
{
    if(currentRow >= label_prob.size())
        return false;
    tipl::out() << "reader header information from " << param.image_file_name[currentRow];
    tipl::io::gz_nifti in;
    if(!in.load_from_file(param.image_file_name[currentRow]))
    {
        error_msg = in.error_msg;
        return false;
    }
    tipl::out() << "save " << file_name;
    in.flip_swap_seq = raw_image_flip_swap[currentRow];
    if(is_label[currentRow])
    {
        tipl::image<3,unsigned char> label(label_prob[currentRow]);
        in.apply_flip_swap_seq(label,true);

        if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
            in.load_from_image(label);
        else
            in.load_from_image(label.alias(0,tipl::shape<4>(
                          raw_image_shape[currentRow][0],
                          raw_image_shape[currentRow][1],
                          raw_image_shape[currentRow][2],
                          label_prob[currentRow].depth()/raw_image_shape[currentRow][2])));
        return in.save_to_file(file_name);
    }

    tipl::image<3> prob(label_prob[currentRow]);
    in.apply_flip_swap_seq(prob,true);

    if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
        in.load_from_image(prob);
    else
        in.load_from_image(prob.alias(0,tipl::shape<4>(
                          raw_image_shape[currentRow][0],
                          raw_image_shape[currentRow][1],
                          raw_image_shape[currentRow][2],
                          label_prob[currentRow].depth()/raw_image_shape[currentRow][2])));
    return in.save_to_file(file_name);
}



