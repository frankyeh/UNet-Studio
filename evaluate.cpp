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
    eval.resize(param.image_file_name.size());
    data_ready = std::vector<bool> (param.image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        for(size_t i = 0;i < eval.size() && !aborted;++i)
        {
            eval[i].model_dim = model->dim;
            eval[i].model_vs = model->voxel_size;
            eval[i].in_count = model->in_count;
            eval[i].out_count = model->out_count;

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
                return error_msg = in.error_msg,aborted = true,void();
            if(in.dim(4) != model->in_count)
                return error_msg = param.image_file_name[i] + " has inconsistent input channel",aborted = true,void();


            in.get_image_transformation(eval[i].untouched_srow);

            tipl::image<3> raw_image;
            in >> raw_image >> eval[i].image_vs >> eval[i].srow;

            eval[i].flip_swap = in.flip_swap_seq;
            eval[i].image_dim = raw_image.shape();

            tipl::out() << "channel:" << in.dim(4) << " dim: " << raw_image.shape() << " vs:" << eval[i].image_vs;
            tipl::out() << "adjust intensity by normalizing otsu median value";

            if(model->in_count > 1)
            {
                raw_image.resize(eval[i].image_dim.multiply(tipl::shape<3>::z,model->in_count));
                tipl::out() << "handle multiple channels. model channel count:" << model->in_count;
                for(size_t c = 1;c < model->in_count;++c)
                {
                    auto image = raw_image.alias(c*eval[i].image_dim.size(),eval[i].image_dim);
                    if(!(in >> image))
                    {
                        error_msg = param.image_file_name[i] + " reading failed";
                        aborted = true;
                        return;
                    }
                }
            }
            tipl::out() << "preprocessing";
            if(!eval[i].preproc_actions(raw_image))
                return error_msg = "invalid image for processing: " + param.image_file_name[i],aborted = true,void();
            data_ready[i] = true;
        }
    }));
}

void evaluate_unet::evaluate(void)
{
    evaluate_thread.reset(new std::thread([=](){
        try{
            for (cur_prog = 0; cur_prog < eval.size() && !aborted; cur_prog++)
            {
                while(!data_ready[cur_prog])
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    if(aborted)
                        return;
                    status = "preproc_actions";
                }
                tipl::out() << "inferencing using u-net";
                eval[cur_prog].model_output.resize(eval[cur_prog].model_input.size());
                torch::NoGradGuard no_grad;
                for(size_t i = 0;i < eval[cur_prog].model_input.size();++i)
                {
                    auto& cur_input = eval[cur_prog].model_input[i];
                    auto& cur_output = eval[cur_prog].model_output[i];
                    cur_output.resize(cur_input.shape().multiply(tipl::shape<3>::z,model->out_count).divide(tipl::shape<3>::z,model->in_count));
                    auto out = model->forward(torch::from_blob(cur_input.data(),
                                              {1,model->in_count,int(cur_input.depth()/model->in_count),int(cur_input.height()),int(cur_input.width())}).to(param.device))[0];
                    std::memcpy(cur_output.data(),out.to(torch::kCPU).contiguous().data_ptr<float>(),cur_output.size()*sizeof(float));
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
    char is_label;
    tipl::ml3d::postproc_actions(cmd,param1,param2,eval[cur_output].label_prob,eval[cur_output].fg_prob,eval[cur_output].image_dim,is_label);
}



extern std::vector<std::string> seg_template_list;
extern std::vector<std::vector<std::string> > atlas_file_name_list;;
void evaluate_unet::output(void)
{
    auto run_evaluation = [this]()
    {
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false;}
        } guard(running);

        tipl::image<3> gaussian_weight(model->dim);
        tipl::ml3d::create_gaussian(gaussian_weight);


        try{
            for (cur_output = 0;cur_output < eval.size() && !aborted; cur_output++)
            {
                while(cur_output >= cur_prog)
                {
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    status = "evaluating";
                }
                if(eval[cur_output].model_output.empty())
                    continue;

                eval[cur_output].get_label_prob(gaussian_weight);
                eval[cur_output].remove_bg_channel();
                eval[cur_output].create_mask(param.prob_threshold);


                auto& cur_foreground_prob = eval[cur_output].fg_prob;
                auto& cur_label_prob = eval[cur_output].label_prob;

                if(aborted)
                    return;

                switch(proc_strategy.output_format)
                {
                    case 0: // 3D label

                        eval[cur_output].get_label(param.prob_threshold);

                        if(!template_I.empty())
                        {
                            const auto& cur_shape = eval[cur_output].image_dim;
                            const size_t max_tissue_count = 4;/* exclude csf */
                            auto cur_label = tipl::image<3,unsigned char>(cur_label_prob);
                            tipl::reg::mm_reg<tipl::out> reg;
                            tipl::progress prog("register to template");
                            tipl::expand_label_to_images(template_I,reg.It,max_tissue_count);
                            tipl::expand_label_to_images(cur_label,reg.I,max_tissue_count);
                            reg.ItR = template_R;
                            reg.IR = eval[cur_output].srow;
                            reg.Itvs = template_vs;
                            reg.Ivs = eval[cur_output].image_vs;
                            reg.Its = template_I.shape();
                            reg.Is = cur_shape;
                            reg.match_resolution(true);
                            reg.param.speed = 1.0f;
                            reg.param.smoothing = 0.0f;

                            reg.linear_reg(aborted);
                            reg.nonlinear_reg(aborted);
                            reg.to_It_space(template_I.shape(),template_R);
                            reg.to_I_space(cur_shape,eval[cur_output].srow);
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
                            cur_label_prob = atlas;
                        }
                        break;
                    case 1: // skull strip
                    case 2: // mask
                        {
                            tipl::filter::gaussian(cur_foreground_prob);
                            tipl::filter::gaussian(cur_foreground_prob);

                            if(proc_strategy.output_format == 2) // mask
                            {
                                cur_label_prob = cur_foreground_prob;
                                break;
                            }
                            tipl::image<3> I;
                            if(!(tipl::io::gz_nifti(param.image_file_name[cur_output],std::ios::in) >> I))
                            {
                                tipl::error() << "cannot read image file:" << param.image_file_name[cur_output];
                                cur_label_prob.clear();
                                break;
                            }
                            for(size_t pos = 0,sz = std::min<size_t>(I.size(),cur_foreground_prob.size());pos < sz;++pos)
                                I[pos] *= cur_foreground_prob[pos];
                            tipl::normalize(I);
                            cur_label_prob.swap(I);
                        }
                    break;

                }

                eval[cur_output].model_input.clear();
                eval[cur_output].model_output.clear();

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
    if(currentRow >= eval.size())
        return false;
    return eval[currentRow].save_to_file<tipl::io::gz_nifti>(file_name);
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

    eval.param.prob_threshold = po.get("prob_threshold",eval.param.prob_threshold);
    eval.proc_strategy.output_format = po.get("output_format",eval.proc_strategy.output_format);
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
