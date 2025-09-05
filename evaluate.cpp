#include "evaluate.hpp"

extern tipl::program_option<tipl::out> po;

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


template<int dim>
inline auto subject_image_pre(tipl::image<dim>&& I)
{
    tipl::image<dim,unsigned char> out;
    tipl::filter::gaussian(I);
    tipl::segmentation::normalize_otsu_median(I);
    tipl::normalize_upper_lower2(I,out,255.999f);
    return out;
}
template<int dim>
inline auto subject_image_pre(const tipl::image<dim>& I)
{
    return subject_image_pre(tipl::image<dim>(I));
}

inline size_t linear_with_mi(const tipl::image<3,float>& from,
                            const tipl::vector<3>& from_vs,
                            const tipl::image<3,float>& to,
                            const tipl::vector<3>& to_vs,
                              tipl::affine_transform<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float bound[3][8] = tipl::reg::reg_bound)
{
    return tipl::reg::linear<tipl::out>(tipl::reg::make_list(subject_image_pre(from)),from_vs,
                                        tipl::reg::make_list(subject_image_pre(to)),to_vs,
                                        arg,reg_type,terminated,bound);
}

inline void rotate_to_template(tipl::image<3>& images,
                         const tipl::shape<3>& image_dim,
                         const tipl::vector<3>& image_vs,
                         const tipl::image<3>& template_image,
                         const tipl::vector<3>& template_image_vs,
                         const tipl::shape<3>& model_dim,
                         const tipl::vector<3>& model_vs,
                         tipl::transformation_matrix<float>& trans)
{
    tipl::out() << "rotate to template";
    tipl::affine_transform<float> arg_rotated;
    auto image0 = images.alias(0,image_dim);
    bool terminated = false;
    linear_with_mi(template_image,template_image_vs,image0,image_vs,arg_rotated,
                                  tipl::reg::rigid_body,terminated,tipl::reg::large_bound);
    trans = tipl::transformation_matrix<float>(arg_rotated,model_dim,model_vs,image_dim,image_vs);

    int in_channel = images.depth()/image_dim[2];
    tipl::image<3> target_images(model_dim.multiply(tipl::shape<3>::z,in_channel));
    tipl::par_for(in_channel,[&](int c)
    {
        auto image = images.alias(image_dim.size()*c,image_dim);
        auto target_image = target_images.alias(model_dim.size()*c,model_dim);
        tipl::resample(image,target_image,trans);
        tipl::normalize(target_image);
    });
    target_images.swap(images);
    tipl::lower_threshold(images,0.0f);
}



void postproc_actions(const std::string& command,
                      float param1,float param2,
                      tipl::image<3>& this_image,
                      tipl::image<3>& prob,
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
        tipl::adaptive_par_for(dim.size(),[&](size_t pos)
        {
            float m = 0.0f;
            for(size_t i = pos;i < this_image.size();i += dim.size())
                if(this_image[i] > m)
                    m = this_image[i];
            if(prob[pos] <= soft_min_prob)
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
        tipl::adaptive_par_for(dim.size(),[&](size_t pos)
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

void evaluate_unet::read_file(void)
{
    evaluate_input = std::vector<tipl::image<3> >(param.image_file_name.size());
    raw_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    raw_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    raw_image_trans = std::vector<tipl::transformation_matrix<float> >(param.image_file_name.size());
    raw_image_flip_swap = std::vector<std::vector<char> >(param.image_file_name.size());
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
            tipl::image<3> raw_image;
            in >> raw_image;
            in.get_voxel_size(raw_image_vs[i]);
            raw_image_flip_swap[i] = in.flip_swap_seq;
            raw_image_shape[i] = raw_image.shape();

            tipl::out() << "channel:" << in.dim(4) << "dim: " << raw_image.shape() << " vs:" << raw_image_vs[i] << std::endl;

            tipl::segmentation::normalize_otsu_median(raw_image);
            tipl::out() << "adjust intensity by normalizing otsu median value";

            evaluate_input[i] = raw_image;
            evaluate_input[i].resize(raw_image_shape[i].multiply(tipl::shape<3>::z,model->in_count));

            if(model->in_count)
            {
                tipl::out() << "handle multiple channels. model channel count:" << model->in_count;
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
                    tipl::segmentation::normalize_otsu_median(evaluate_input[i]);
                }
            }

            if(!template_image.empty())
                rotate_to_template(evaluate_input[i],
                                   raw_image_shape[i],
                                   raw_image_vs[i],
                                   template_image,template_image_vs,
                                   model->dim,model->voxel_size,
                                   raw_image_trans[i]);
            else
                tipl::ml3d::preproc_actions(evaluate_input[i],
                                raw_image_shape[i],
                                raw_image_vs[i],
                                model->dim,model->voxel_size,
                                raw_image_trans[i],
                                proc_strategy.match_resolution,proc_strategy.match_fov);
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

                evaluate_output[cur_prog].resize(cur_input.shape().multiply(tipl::shape<3>::z,model->out_count).divide(tipl::shape<3>::z,model->in_count));
                tipl::out() << "input dimension:" << cur_input.shape() << " vs:" << raw_image_vs[cur_prog];
                if((cur_input.width()/2 >= model->dim.width() ||
                   raw_image_vs[cur_prog][0] * 2.0f <= model->voxel_size[0]) && model->in_count == 1)
                {
                    tipl::out() << "subsample by 2x to handle large image volume";
                    for(size_t i = 0;i < 8;++i)
                    {
                        tipl::image<3,float> subI,result;
                        subsample8(cur_input,subI,i);
                        tipl::out() << "inferencing using u-net at subsample " << i;
                        auto out = model->forward(torch::from_blob(subI.data(),{1,1,int(subI.depth()),int(subI.height()),int(subI.width())}).to(param.device));
                        result.resize(subI.shape().multiply(tipl::shape<3>::z,model->out_count));
                        std::memcpy(result.data(),out.to(torch::kCPU).data_ptr<float>(),result.size()*sizeof(float));
                        for(size_t j = 0;j < model->out_count;++j)
                        {
                            auto eval_output = tipl::make_image(evaluate_output[cur_prog].data() + j*cur_input.size(),cur_input.shape());
                            upsample8(eval_output,tipl::make_image(result.data() + j*subI.size(),subI.shape()),i);
                        }
                    }
                }
                else
                {
                    tipl::out() << "inferencing using u-net";
                    auto out = model->forward(torch::from_blob(cur_input.data(),
                                              {1,model->in_count,int(cur_input.depth()/model->in_count),int(cur_input.height()),int(cur_input.width())}).to(param.device));
                    std::memcpy(evaluate_output[cur_prog].data(),out.to(torch::kCPU).data_ptr<float>(),evaluate_output[cur_prog].size()*sizeof(float));
                }
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

template<typename T,typename U,typename V>
inline void postproc_actions_Knn(T& label_prob,
                             T& fg_prob,
                             const U& eval_input,
                             const U& eval_output,
                             const V& raw_image,
                             tipl::transformation_matrix<float,3> trans,
                             size_t model_out_count,float prob_threshold)
{
    tipl::shape<3> dim_from(eval_output.shape().divide(tipl::shape<3>::z,model_out_count)),
                   dim_to(raw_image.shape());
    label_prob.resize(dim_to.multiply(tipl::shape<3>::z,model_out_count));
    trans.inverse();
    tipl::par_for(model_out_count,[&](int i)
    {
        auto from_out = eval_output.alias(dim_from.size()*i,dim_from);
        auto to = label_prob.alias(dim_to.size()*i,dim_to);
        for(tipl::pixel_index<3> index(dim_to);index < dim_to.size();++index)
        {
            auto cur_v = dim_to[index.index()];
            tipl::vector<3> pos;
            trans(index,pos);
            pos.round();
            auto input = tipl::get_window(tipl::pixel_index<3>(pos[0],pos[1],pos[2],eval_input.shape()),eval_input,2);
            for(auto& each : input)
                each = std::fabs(each-cur_v);
            auto output = tipl::get_window(tipl::pixel_index<3>(pos[0],pos[1],pos[2],from_out.shape()),from_out,2);
            std::vector<size_t> p(input.size());
            std::iota(p.begin(), p.end(), 0);
            std::sort(p.begin(), p.end(),[&](size_t i, size_t j){ return input[i] < input[j]; });
            std::vector<float> sorted_output(p.size());
            for (size_t i = 0; i < p.size(); ++i)
                sorted_output[i] = output[p[i]];
            to[index.index()] = tipl::mean(sorted_output.begin(),
                                           sorted_output.begin() + std::min<size_t>(sorted_output.size()/2,5));
        }
        tipl::preserve(to.begin(),to.end(),raw_image.begin());

    },model_out_count);
    auto I = tipl::make_image(label_prob.data(),dim_to.expand(label_prob.depth()/dim_to[2]));
    fg_prob = tipl::ml3d::defragment4d(I,prob_threshold);
}

void evaluate_unet::output(void)
{
    label_prob = std::vector<tipl::image<3> >(param.image_file_name.size());
    foreground_prob = std::vector<tipl::image<3> >(param.image_file_name.size());
    is_label = std::vector<char>(param.image_file_name.size());
    auto run_evaluation = [this]()
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

                tipl::ml3d::postproc_actions(label_prob[cur_output],
                                             evaluate_output[cur_output],
                                             raw_image_shape[cur_output],
                                             raw_image_trans[cur_output],
                                             model->out_count,!proc_strategy.match_fov && !proc_strategy.match_resolution);

                auto label_prob_4d = tipl::make_image(label_prob[cur_output].data(),raw_image_shape[cur_output].expand(model->out_count));
                foreground_prob[cur_output] = tipl::ml3d::defragment4d(label_prob_4d,param.prob_threshold);

                if(aborted)
                    return;


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
                    case 3: // mask
                        label_prob[cur_output] = foreground_prob[cur_output];
                        tipl::upper_threshold(label_prob[cur_output],param.prob_threshold);
                        tipl::filter::gaussian(label_prob[cur_output]);
                        tipl::normalize(label_prob[cur_output]);
                    break;

                }

                evaluate_input[cur_output] = tipl::image<3>();
                evaluate_output[cur_output] = tipl::image<3>();

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
    };

    if(tipl::show_prog)
        output_thread.reset(new std::thread(run_evaluation));
    else
    {
        run_evaluation();
        join();
    }
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
    tipl::vector<3> vs;
    in.get_image_transformation(trans);
    out.set_image_transformation(trans);
    in.get_voxel_size(vs);
    out.set_voxel_size(vs);

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

bool load_from_file(UNet3d& model,const char* file_name);
std::string get_network_path(void);
int eval(void)
{
    static evaluate_unet eval;
    if(eval.running)
    {
        tipl::out() << "terminating training...";
        eval.stop();
    }


    // loading images data
    {
        eval.param.image_file_name.clear();
        if(!po.check("source"))
            return 1;
        if(!po.get_files("source",eval.param.image_file_name))
        {
            tipl::out() << "ERROR: " << eval.error_msg;
            return 1;
        }
    }

    auto network = get_network_path();
    {
        if(!std::filesystem::exists(network))
        {
            tipl::out() << "ERROR: cannot find the network file " << network;
            return 1;
        }
        tipl::out() << "loading network " << network;
        if(!load_from_file(eval.model,network.c_str()))
        {
            tipl::out() << "ERROR: failed to load model from " << network;
            return 1;
        }
        tipl::out() << eval.model->get_info();
    }

    eval.param.prob_threshold = po.get("prob_threshold",0.5f);
    eval.proc_strategy.match_resolution = po.get("match_resolution",1);
    eval.proc_strategy.match_fov = po.get("match_fov",0);
    eval.proc_strategy.match_orientation = po.get("match_orientation",0);
    eval.proc_strategy.output_format = po.get("output_format",0);


    eval.param.device = torch::Device(po.get("device",torch::hasCUDA() ? "cuda:0" :
                                                       (torch::hasHIP() ? "hip:0" :
                                                       (torch::hasMPS() ? "mps:0": "cpu"))));
    {
        tipl::progress p("start evaluating");
        eval.start();
        eval.join();
    }
    if(!eval.error_msg.empty())
    {
        tipl::out() << "ERROR: " << eval.error_msg;
        return 1;
    }
    {
        tipl::progress p("saving results");
        for(size_t i = 0;p(i,eval.param.image_file_name.size());++i)
        {
            auto file_name = eval.param.image_file_name[i] + ".result.nii.gz";
            tipl::out() << "save to " << file_name;
            if(!eval.save_to_file(i,file_name.c_str()))
                tipl::out() << "ERROR: " << eval.error_msg;
        }
    }

    return 0;
}
