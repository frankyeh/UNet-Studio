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

size_t linear_cuda(const tipl::image<3,float>& from,
                              tipl::vector<3> from_vs,
                              const tipl::image<3,float>& to,
                              tipl::vector<3> to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound);

inline size_t linear_with_mi(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound = tipl::reg::reg_bound)
{
    float result = 0.0f;
    if constexpr (tipl::use_cuda)
        result = linear_cuda(from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),terminated,bound);
    if(result == 0.0f)
        result = tipl::reg::linear_mr<tipl::reg::mutual_information>
                (from,from_vs,to,to_vs,arg,tipl::reg::reg_type(reg_type),[&](void){return terminated;},
                    0.01,bound != tipl::reg::narrow_bound,bound);
    return result;
}

void preproc_actions(tipl::image<3>& images,
                     const tipl::shape<3>& image_dim,
                     const tipl::vector<3>& image_vs,
                     const tipl::image<3>& template_image,
                     const tipl::vector<3>& template_image_vs,
                     tipl::transformation_matrix<float>& trans,
                     const tipl::shape<3>& model_dim,
                     const tipl::vector<3>& model_vs,
                     const ProcStrategy& proc_strategy,
                     std::string& error_msg)
{
    int in_channel = images.depth()/image_dim[2];
    if(model_dim == image_dim && image_vs == model_vs)
    {
        tipl::out() << "image resolution and dimension are the same as training data. No padding or regrinding needed.";
        return;
    }
    auto target_vs = proc_strategy.match_resolution || proc_strategy.match_orientation ? model_vs : image_vs;
    auto target_dim = proc_strategy.match_fov || proc_strategy.match_orientation ? model_dim :
                            unet_inputsize(tipl::shape<3>(float(image_dim.width())*image_vs[0]/target_vs[0],
                                float(image_dim.height())*image_vs[1]/target_vs[1],
                                float(image_dim.depth())*image_vs[2]/target_vs[2]));

    tipl::image<3> target_images(target_dim.multiply(tipl::shape<3>::z,in_channel));
    if(!proc_strategy.match_fov && !proc_strategy.match_resolution)
    {
        auto shift = tipl::vector<3,int>(target_dim)-tipl::vector<3,int>(image_dim);
        shift[0] /= 2;
        shift[1] /= 2;
        tipl::par_for(in_channel,[&](int c)
        {
            auto image = images.alias(image_dim.size()*c,image_dim);
            auto target_image = target_images.alias(target_dim.size()*c,target_dim);
            tipl::draw(image,target_image,shift);
            tipl::normalize(target_image);
        });

    }
    else
    {
        tipl::affine_transform<float> arg;
        if(proc_strategy.match_orientation && !template_image.empty())
        {
            auto image0 = images.alias(0,image_dim);
            tipl::out() << "rotating images to template orientation" << std::endl;
            auto arg_rotated = arg;
            bool terminated = false;
            linear_with_mi(template_image,template_image_vs,image0,image_vs,arg_rotated,
                                          tipl::reg::rigid_body,terminated,tipl::reg::large_bound);
            trans = tipl::transformation_matrix<float>(arg_rotated,target_dim,target_vs,image_dim,image_vs);
        }
        else
            trans = tipl::transformation_matrix<float>(arg,target_dim,target_vs,image_dim,image_vs);

        tipl::par_for(in_channel,[&](int c)
        {
            auto image = images.alias(image_dim.size()*c,image_dim);
            auto target_image = target_images.alias(target_dim.size()*c,target_dim);
            tipl::resample_mt(image,target_image,trans);
            tipl::normalize(target_image);
        });

    }
    target_images.swap(images);
}





void postproc_actions(const std::string& command,
                      float param1,float param2,
                      tipl::image<3>& this_image,
                      tipl::image<3>& foreground_prob,
                      const tipl::shape<3>& dim,
                      char& is_label)
{
    auto this_image_frames = this_image.depth()/dim[2];
    if(this_image.empty())
        return;
    tipl::out() << "run " << command;
    if(command == "upper_threshold")
    {
        float upper_threshold_threshold = param1;
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::upper_threshold(I,upper_threshold_threshold);
        });
        is_label = false;
        return;
    }
    if(command == "lower_threshold")
    {
        float lower_threshold_threshold = param1;
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::lower_threshold(I,lower_threshold_threshold);
        });
        is_label = false;
        return;
    }
    if(command == "minus")
    {
        float minus_value = param1;
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            for(size_t i = 0;i < I.size();++i)
                I[i] -= minus_value;
        });
        is_label = false;
        return;
    }

    if(command == "defragment_each")
    {
        float defragment_each_threshold = param1;
        tipl::par_for(this_image_frames,[&](size_t label)
        {
            auto I = this_image.alias(dim.size()*label,dim);
            tipl::image<3,char> mask(I.shape()),mask2;
            for(size_t i = 0;i < I.size();++i)
                mask[i] = (I[i] > defragment_each_threshold ? 1:0);
            mask2 = mask;
            tipl::morphology::defragment_by_size_ratio(mask);
            for(size_t i = 0;i < I.size();++i)
                if(!mask[i] && mask[2])
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
    if(command =="soft_max")
    {
        float soft_min_prob = param1;
        tipl::par_for(dim.size(),[&](size_t pos)
        {
            float m = 0.0f;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                if(this_image[i] > m)
                    m = this_image[i];
            if(foreground_prob[pos] <= soft_min_prob)
            {
                for(size_t i = pos;i < this_image.size();i += dim.size())
                    this_image[i] = 0.0f;
                return;
            }
            for(size_t i = pos;i < this_image.size();i += dim.size())
                this_image[i] = (this_image[i] >= m ? 1.0f:0.0f);
        });
        is_label = true;
        return;
    }
    if(command =="to_3d_label")
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
    evaluate_input = std::vector<tipl::image<3> >(param.image_file_name.size());
    raw_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    raw_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    raw_image_trans = std::vector<tipl::transformation_matrix<float> >(param.image_file_name.size());
    raw_image_flip_swap = std::vector<std::vector<char> >(param.image_file_name.size());
    raw_image_mask = std::vector<tipl::image<3,char> >(param.image_file_name.size());
    data_ready = std::vector<bool> (param.image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        tipl::image<3> template_image;
        tipl::vector<3> template_image_vs;
        if(proc_strategy.match_resolution && !proc_strategy.template_file_name.empty())
        {
            tipl::io::gz_nifti in;
            if(in.load_from_file(proc_strategy.template_file_name))
            {
                in >> template_image;
                in.get_voxel_size(template_image_vs);
            }
            else
                tipl::out() << "cannot read template file: " << proc_strategy.template_file_name << std::endl;
        }
        for(size_t i = 0;i < evaluate_input.size() && !aborted;++i)
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
            if(in.dim(4) != model->in_count)
            {
                error_msg = param.image_file_name[i];
                error_msg += " has inconsistent input channel";
                aborted = true;
                return;
            }
            in >> evaluate_input[i];
            tipl::threshold(evaluate_input[i],raw_image_mask[i],0);
            in.get_voxel_size(raw_image_vs[i]);
            raw_image_flip_swap[i] = in.flip_swap_seq;
            raw_image_shape[i] = evaluate_input[i].shape();
            tipl::out() << "channel:" << in.dim(4) << "dim: " << evaluate_input[i].shape() << " vs:" << raw_image_vs[i] << std::endl;

            // handle multiple channels
            evaluate_input[i].resize(raw_image_shape[i].multiply(tipl::shape<3>::z,model->in_count));
            for(size_t c = 1;c < model->in_count;++c)
            {
                auto image = evaluate_input[i].alias(c*raw_image_shape[i].size(),raw_image_shape[i]);
                if(!(in >> image))
                {
                    error_msg = param.image_file_name[i];
                    error_msg += " reading failed";
                    aborted = true;
                    return;
                }
            }
            preproc_actions(evaluate_input[i],
                            raw_image_shape[i],
                            raw_image_vs[i],
                            template_image,template_image_vs,
                            raw_image_trans[i],
                            model->dim,model->voxel_size,
                            proc_strategy,error_msg);
            tipl::lower_threshold(evaluate_input[i],0.0f);
            data_ready[i] = true;
        }
    }));
}

void evaluate_unet::evaluate(void)
{
    evaluate_output  = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_thread.reset(new std::thread([=](){
        try{
            for (cur_prog = 0; cur_prog < evaluate_input.size() && !aborted; cur_prog++)
            {
                auto& cur_input = evaluate_input[cur_prog];
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
                                          {1,model->in_count,int(cur_input.depth()/model->in_count),int(cur_input.height()),int(cur_input.width())}).to(param.device));
                evaluate_output[cur_prog].resize(cur_input.shape().multiply(tipl::shape<3>::z,model->out_count).divide(tipl::shape<3>::z,model->in_count));
                std::memcpy(&evaluate_output[cur_prog][0],out.to(torch::kCPU).data_ptr<float>(),evaluate_output[cur_prog].size()*sizeof(float));
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
                 foreground_prob[cur_output],
                 raw_image_shape[cur_output],
                 is_label[cur_output]);
}
void evaluate_unet::output(void)
{
    label_prob = std::vector<tipl::image<3> >(param.image_file_name.size());
    foreground_prob = std::vector<tipl::image<3> >(param.image_file_name.size());
    is_label = std::vector<char>(param.image_file_name.size());
    output_thread.reset(new std::thread([this]()
    {
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false;}
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
                if(evaluate_output[cur_output].empty())
                    continue;
                tipl::shape<3> dim_from(evaluate_output[cur_output].shape().divide(tipl::shape<3>::z,model->out_count)),
                               dim_to(raw_image_shape[cur_output]);
                label_prob[cur_output].resize(dim_to.multiply(tipl::shape<3>::z,model->out_count));
                tipl::par_for(model->out_count,[&](int i)
                {
                    auto from = evaluate_output[cur_output].alias(dim_from.size()*i,dim_from);
                    auto to = label_prob[cur_output].alias(dim_to.size()*i,dim_to);
                    if(!proc_strategy.match_fov && !proc_strategy.match_resolution)
                    {
                        auto shift = tipl::vector<3,int>(to.shape())-tipl::vector<3,int>(from.shape());
                        shift[0] /= 2;
                        shift[1] /= 2;
                        tipl::draw(from,to,shift);
                    }
                    else
                    {
                        auto trans = raw_image_trans[cur_output];
                        trans.inverse();
                        tipl::resample_mt(from,to,trans);
                    }
                    for(size_t j = 0;j < raw_image_mask[cur_output].size();++j)
                        if(!raw_image_mask[cur_output][j])
                            to[j] = 0;
                });
                if(aborted)
                    return;

                auto I = tipl::make_image(&label_prob[cur_output][0],
                                    raw_image_shape[cur_output].expand(
                                    label_prob[cur_output].depth()/raw_image_shape[cur_output][2]));
                foreground_prob[cur_output] = tipl::ml3d::defragment4d(I,param.prob_threshold);


                switch(proc_strategy.output_format)
                {
                    case 0: // 3D label
                        proc_actions("soft_max",param.prob_threshold);
                        proc_actions("to_3d_label");
                    break;
                    case 2: // skull strip
                        {
                            tipl::image<3> I;
                            tipl::io::gz_nifti in;
                            tipl::image<3> foreground_mask;
                            tipl::threshold(foreground_prob[cur_output],foreground_mask,param.prob_threshold,1,0);
                            tipl::filter::gaussian(foreground_mask);

                            if(in.load_from_file(param.image_file_name[cur_output]))
                            {
                                in >> I;
                                for(size_t pos = 0;pos < I.size() && pos < foreground_mask.size();++pos)
                                    I[pos] *= foreground_mask[pos];
                            }
                            tipl::normalize(I);
                            label_prob[cur_output].swap(I);
                        }
                    break;

                }

                evaluate_input[cur_output] = tipl::image<3>();
                evaluate_output[cur_output] = tipl::image<3>();
                raw_image_mask[cur_output] = tipl::image<3,char>();
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
    tipl::io::gz_nifti in,out;
    if(!in.load_from_file(param.image_file_name[currentRow]))
    {
        error_msg = in.error_msg;
        return false;
    }
    tipl::matrix<4,4,float> trans;
    in.get_image_transformation(trans);
    out.set_image_transformation(trans);

    tipl::out() << "save " << file_name;
    in.flip_swap_seq = raw_image_flip_swap[currentRow];
    if(is_label[currentRow])
    {
        tipl::image<3,unsigned char> label(label_prob[currentRow]);
        in.apply_flip_swap_seq(label,true);
        if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
            out.load_from_image(label);
        else
            out.load_from_image(label.alias(0,tipl::shape<4>(
                          raw_image_shape[currentRow][0],
                          raw_image_shape[currentRow][1],
                          raw_image_shape[currentRow][2],
                          label_prob[currentRow].depth()/raw_image_shape[currentRow][2])));
        return out.save_to_file(file_name);
    }

    tipl::image<3> prob(label_prob[currentRow]);
    in.apply_flip_swap_seq(prob,true);

    if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
        out.load_from_image(prob);
    else
        out.load_from_image(prob.alias(0,tipl::shape<4>(
                          raw_image_shape[currentRow][0],
                          raw_image_shape[currentRow][1],
                          raw_image_shape[currentRow][2],
                          label_prob[currentRow].depth()/raw_image_shape[currentRow][2])));
    return out.save_to_file(file_name);
}



