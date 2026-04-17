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
                              tipl::affine_param<float>& arg,
                              tipl::reg::reg_type reg_type,
                              bool& terminated,
                              const float* bound);



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
    tipl::affine_param<float> arg_rotated;
    auto image0 = images.alias(0,image_dim);
    bool terminated = false;
    tipl::reg::linear<tipl::out>(tipl::reg::make_list(tipl::reg::template_image_pre(template_image)),template_image_vs,
                                 tipl::reg::make_list(tipl::reg::subject_image_pre(tipl::image<3>(image0))),image_vs,
                                            arg_rotated,{tipl::reg::rigid_body,tipl::reg::mutual_info,tipl::reg::large_bound},terminated);
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
            for(size_t i = 0,sz = I.size();i < sz;++i)
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
            for(size_t i = 0,sz = I.size();i < sz;++i)
                mask[i] = (I[i] > defragment_each_threshold ? 1:0);
            mask2 = mask;
            tipl::morphology::defragment_by_size_ratio(mask);
            for(size_t i = 0,sz = I.size();i < sz;++i)
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
    tipl::error() << "unknown command " << command << std::endl;
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


bool evaluate_unet::load_atlas(const std::string& file_name)
{
    std::string corrected_file_name;
    {
        std::filesystem::path corrected_dir = std::filesystem::path(QCoreApplication::applicationDirPath().toUtf8().constData())/"corrected_atlas";
        if(!std::filesystem::exists(corrected_dir))
            std::filesystem::create_directories(corrected_dir);
        corrected_file_name = (corrected_dir/(tipl::remove_all_suffix(std::filesystem::path(file_name).filename().string()) + ".corrected.nii.gz")).string();
    }

    atlas_I.resize(template_I.shape());
    if(std::filesystem::exists(corrected_file_name))
        return tipl::io::gz_nifti(corrected_file_name,std::ios::in).to_space<tipl::majority>(atlas_I,template_R) >> [&](const std::string& e){error_msg = e;};

    if(!(tipl::io::gz_nifti(file_name,std::ios::in).to_space<tipl::majority>(atlas_I,template_R) >> [&](const std::string& e){error_msg = e;}))
        return false;

    // in template_I is the tissue segmentation, 0 is background, 1: white matter 2: gray matter 3: cerebellar gray matter 4: subcortical
    const size_t template_region_count = 5;
    atlas_region_count = tipl::max_value(atlas_I);

    // zero atlas_I where template_I is zero
    tipl::preserve(atlas_I.begin(),atlas_I.end(),template_I.begin());

    std::vector<size_t> tissue_total;
    tipl::histogram(template_I,tissue_total,0,template_region_count,template_region_count);



    std::vector<float> tissue_coverage(template_region_count,0.0f);
    {
        tipl::progress prog("checking tissue coverage of the atlas");
        std::vector<size_t> atlas_covered(template_region_count,0);
        for(size_t pos = 0,sz = atlas_I.size();pos < sz;++pos)
            if(atlas_I[pos] > 0 && template_I[pos] < template_region_count)
                ++atlas_covered[template_I[pos]];
        for(size_t cur_tissue = 1;cur_tissue < template_region_count;++cur_tissue)
        {
            if(tissue_total[cur_tissue] == 0)
                continue;
            tissue_coverage[cur_tissue] = float(atlas_covered[cur_tissue])/float(tissue_total[cur_tissue]);
            tipl::out() << tissue_names[cur_tissue] << " coverage: " << int(tissue_coverage[cur_tissue]*100.0f) << "%";
        }
    }

    {
        tipl::progress prog("checking tissue classification of each atlas region");
        tipl::morphology::reclassify_labels_by_template<tipl::out>(template_I,atlas_I);
    }

    {
        tipl::progress prog("region growing to make up missing voxels");
        tipl::image<3,unsigned char> mask(template_I.shape());

        for(size_t cur_tissue = 1;cur_tissue < template_region_count;++cur_tissue)
        {
            if(tissue_coverage[cur_tissue] <= 0.75f)
                continue;
            tipl::out() << "work on " << tissue_names[cur_tissue];
            for(size_t pos = 0,sz = template_I.size();pos < sz;++pos)
                mask[pos] = (template_I[pos] == cur_tissue);
            tipl::morphology::fill_and_smooth_labels<tipl::out>(mask,atlas_I);
        }
    }

    tipl::io::gz_nifti(corrected_file_name,std::ios::out) << template_vs << template_R << true << atlas_I;
    return true;
}

void evaluate_unet::read_file(void)
{
    evaluate_input = std::vector<tipl::image<3> >(param.image_file_name.size());
    raw_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    raw_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    untouched_srow = raw_image_srow = std::vector<tipl::matrix<4,4,float> >(param.image_file_name.size());
    raw_image_trans = std::vector<tipl::transformation_matrix<float> >(param.image_file_name.size());
    raw_image_flip_swap = std::vector<std::vector<char> >(param.image_file_name.size());
    data_ready = std::vector<bool> (param.image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        tipl::image<3> template_image;
        tipl::vector<3> template_image_vs;
        if(proc_strategy.match_resolution && !proc_strategy.template_file_name.empty())
        {
            if(!(tipl::io::gz_nifti(proc_strategy.template_file_name,std::ios::in) >> std::tie(template_image,template_image_vs)))
            {
                tipl::error() << (error_msg = "cannot read template file: " + proc_strategy.template_file_name);
                aborted = true;
                return;
            }
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
            tipl::out() << "reading " << param.image_file_name[i];
            tipl::io::gz_nifti in(param.image_file_name[i],std::ios::in);
            if(!in)
            {
                error_msg = in.error_msg;
                aborted = true;
                return;
            }
            if(in.dim(4) != model->in_count)
            {
                error_msg = param.image_file_name[i] + " has inconsistent input channel";
                aborted = true;
                return;
            }
            in.get_image_transformation(untouched_srow[i]);

            tipl::image<3> raw_image;
            in >> raw_image >> raw_image_vs[i] >> raw_image_srow[i];
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
                        error_msg = param.image_file_name[i] + " reading failed";
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
                        auto out = model->forward(torch::from_blob(subI.data(),{1,1,int(subI.depth()),int(subI.height()),int(subI.width())}).to(param.device))[0];
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
                                              {1,model->in_count,int(cur_input.depth()/model->in_count),int(cur_input.height()),int(cur_input.width())}).to(param.device))[0];
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




extern std::vector<std::string> seg_template_list;
extern std::vector<std::vector<std::string> > atlas_file_name_list;;
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
                const auto& cur_shape = raw_image_shape[cur_output];
                auto& cur_label_prob = label_prob[cur_output];
                auto& cur_foreground_prob = foreground_prob[cur_output];

                tipl::ml3d::postproc(evaluate_output[cur_output],
                                     cur_shape,
                                     raw_image_trans[cur_output],
                                     model->out_count,
                                     true, /* has_bg_channel / deep_supervision */
                                     param.prob_threshold,
                                     cur_label_prob,
                                     cur_foreground_prob);
                if(aborted)
                    return;

                switch(proc_strategy.output_format)
                {
                    case 0: // 3D label
                        // Background removal and tissue probability adjustments were deleted!
                        // The new `postproc_actions` handles it upstream.

                        proc_actions("soft_max",param.prob_threshold);
                        proc_actions("to_3d_label");

                        if(!template_I.empty())
                        {
                            const size_t max_tissue_count = 4;/* exclude csf */
                            auto cur_label = tipl::image<3,unsigned char>(label_prob[cur_output]);
                            tipl::reg::mm_reg<tipl::out> reg;
                            tipl::progress prog("register to template");
                            tipl::expand_label_to_images(template_I,reg.It,max_tissue_count);
                            tipl::expand_label_to_images(cur_label,reg.I,max_tissue_count);
                            reg.ItR = template_R;
                            reg.IR = raw_image_srow[cur_output];
                            reg.Itvs = template_vs;
                            reg.Ivs = raw_image_vs[cur_output];
                            reg.Its = template_I.shape();
                            reg.Is = cur_shape;
                            reg.match_resolution(true);
                            reg.param.speed = 1.0f;
                            reg.param.smoothing = 0.0f;

                            reg.linear_reg(aborted);
                            reg.nonlinear_reg(aborted);
                            reg.to_It_space(template_I.shape(),template_R);
                            reg.to_I_space(cur_shape,raw_image_srow[cur_output]);
                            auto template_J = reg.apply_warping<false,tipl::majority>(template_I);
                            auto atlas = reg.apply_warping<false,tipl::majority>(atlas_I);

                            // apply mask from current segmentation to warpped template and atlas
                            tipl::preserve(atlas.begin(),atlas.end(),cur_label.begin());
                            tipl::preserve(template_J.begin(),template_J.end(),cur_label.begin());

                            size_t size = cur_shape.size();
                            tipl::image<3,unsigned char> mask(cur_shape);
                            for(size_t cur_tissue = 1;cur_tissue <= max_tissue_count;++cur_tissue)
                            {
                                tipl::out() << "checking tissue:" << tissue_names[cur_tissue];
                                size_t total = 0,has_atlas = 0;
                                for(size_t i = 0;i < size;++i)
                                    if(template_J[i] == cur_tissue && atlas[i])
                                    {
                                        ++total;
                                        if(atlas[i])
                                            ++has_atlas;
                                    }

                                if(!has_atlas)
                                {
                                    tipl::out() << tissue_names[cur_tissue] << " has no label in current atlas, assign zeros";
                                    for(size_t i = 0;i < size;++i)
                                        if(template_J[i] == cur_tissue)
                                            atlas[i] = 0;
                                }
                                if(float(has_atlas)/float(total) > 0.75f)
                                {
                                    tipl::out() << tissue_names[cur_tissue] << " has labels, fill empty labels and smooth atlas";
                                    for(size_t i = 0;i < size;++i)
                                        mask[i] = (template_J[i] == cur_tissue);
                                    tipl::morphology::fill_and_smooth_labels<tipl::out>(mask,atlas);
                                }
                            }
                            label_prob[cur_output] = atlas;
                        }
                        break;
                    case 2: // skull strip
                        {
                            tipl::image<3> I;
                            if(!(tipl::io::gz_nifti(param.image_file_name[cur_output],std::ios::in) >> I))
                            {
                                tipl::error() << "cannot read image file:" << param.image_file_name[cur_output];
                                label_prob[cur_output].clear();
                                break;
                            }
                            for(size_t pos = 0,sz = std::min<size_t>(I.size(),cur_foreground_prob.size());pos < sz;++pos)
                                I[pos] *= cur_foreground_prob[pos];
                            tipl::normalize(I);
                            label_prob[cur_output].swap(I);
                        }
                    break;
                    case 3: // mask
                        label_prob[cur_output] = cur_foreground_prob;
                        tipl::upper_threshold(label_prob[cur_output].begin(),label_prob[cur_output].end(),param.prob_threshold);
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
        if(!error_msg.empty())
            tipl::error() << error_msg;
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
    tipl::io::gz_nifti out;
    if(!out.open(file_name,std::ios::out))
        return error_msg = out.error_msg,false;
    out << untouched_srow[currentRow] << raw_image_vs[currentRow];
    tipl::out() << "save " << file_name;
    out.flip_swap_seq = raw_image_flip_swap[currentRow];
    if(is_label[currentRow])
    {
        tipl::image<3,unsigned char> label(label_prob[currentRow]);
        out.apply_flip_swap_seq(label,true);

        if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
            out << label;
        else
            out << label.alias(0,tipl::shape<4>(
                          raw_image_shape[currentRow][0],
                          raw_image_shape[currentRow][1],
                          raw_image_shape[currentRow][2],
                          label_prob[currentRow].depth()/raw_image_shape[currentRow][2]));
    }
    else
    {
        tipl::image<3> prob(label_prob[currentRow]);
        out.apply_flip_swap_seq(prob,true);
        if(label_prob[currentRow].depth() == raw_image_shape[currentRow][2])
            out << prob;
        else
            out << prob.alias(0,tipl::shape<4>(
                              raw_image_shape[currentRow][0],
                              raw_image_shape[currentRow][1],
                              raw_image_shape[currentRow][2],
                              label_prob[currentRow].depth()/raw_image_shape[currentRow][2]));
    }
    return true;
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
    if(!po.check("source"))
        return 1;
    if((eval.param.image_file_name = po.get_files("source")).empty())
        return tipl::error() << "no file specified at --source",1;

    auto network = get_network_path();
    {
        if(!std::filesystem::exists(network))
            return tipl::error() << "cannot find the network file " << network,1;
        tipl::out() << "loading network " << network;
        if(!load_from_file(eval.model,network.c_str()))
            return tipl::error() << "failed to load model from " << network,1;
        tipl::out() << eval.model->get_info();
    }

    if(po.has("template") && po.has("atlas"))
    {
        size_t seg_id = po.get("template",seg_template_list,0);
        if(seg_id >= seg_template_list.size())
            return tipl::error() << "invalid template",1;
        if(!eval.load_template(seg_template_list[seg_id]) ||
           !eval.load_atlas(po.get("atlas",atlas_file_name_list[seg_id],atlas_file_name_list[seg_id][0])))
            return tipl::error() << eval.error_msg,1;
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
        return tipl::error() << eval.error_msg,1;
    tipl::progress p("saving results");
    for(size_t i = 0;p(i,eval.param.image_file_name.size());++i)
    {
        auto file_name = eval.param.image_file_name[i] + ".result.nii.gz";
        tipl::out() << "save to " << file_name;
        if(!eval.save_to_file(i,file_name.c_str()))
            tipl::error() << eval.error_msg;
    }
    return 0;
}
