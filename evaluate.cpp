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
bool preprocessing(tipl::image<3>& image,
                   tipl::vector<3>& image_vs,
                   tipl::matrix<4,4>& image_trans,
                   const tipl::shape<3>& input_dim,
                   const tipl::vector<3>& input_vs,
                   OptionTableWidget* option,
                   unsigned char input_size_strategy,
                   std::string& error_msg)
{

    bool is_mni = false;
    for(int i = 1;i <= 3;++i)
    {
        auto cmd = option->get<int>((std::string("preproc_operation")+std::to_string(i)).c_str());
        if(!cmd)
            continue;
        if(!tipl::command<tipl::io::gz_nifti>(image,image_vs,image_trans,is_mni,operations[cmd],std::string(),error_msg))
            return false;
    }

    if(input_dim == image.shape() && image_vs == input_vs)
    {
        tipl::out() << "image resolution and dimension are the same as training data. No padding or regrinding needed.";
        return true;
    }

    switch(input_size_strategy)
    {
        case 0: //match resolution
        if(image_vs != input_vs)
        {
            tipl::out() << " image has a resolution of " << image_vs << ". regriding to " << input_vs;
            tipl::image<3> new_sized_image(input_dim);
            tipl::resample_mt(image,new_sized_image,
                    tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                        input_dim,input_vs,image.shape(),image_vs));
            new_sized_image.swap(image);
            return true;
        }
        case 2: //original
        {
            tipl::out() << " image has a different dimension. padding or cropping applied";
            tipl::image<3> new_sized_image(input_dim);
            tipl::draw(image,new_sized_image);
            new_sized_image.swap(image);
        }
        return true;
        case 1: //match sizes
        {
            tipl::out() << " image has a dimension of " << image.shape() << ". regriding applied";
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
void evaluate_unet::read_file(const EvaluateParam& param)
{
    network_input = std::vector<tipl::image<3> >(param.image_file_name.size());
    raw_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    raw_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    raw_image_trans2mni = std::vector<tipl::matrix<4,4> >(param.image_file_name.size());
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
            if(!tipl::io::gz_nifti::load_from_file(param.image_file_name[i].c_str(),
                                                  network_input[i],
                                                  raw_image_vs[i],
                                                  raw_image_trans2mni[i]))
            {
                error_msg = "cannot open file ";
                error_msg = param.image_file_name[i];
                aborted = true;
                return;
            }

            raw_image_shape[i] = network_input[i].shape();

            if(!preprocessing(network_input[i],
                          raw_image_vs[i],
                          raw_image_trans2mni[i],
                          model->dim,model->voxel_size,
                          option,input_size_strategy,error_msg))
            {
                aborted = true;
                return;
            }
            data_ready[i] = true;
        }
    }));
}

void evaluate_unet::evaluate(const EvaluateParam& param)
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
                    status = "preprocessing";
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

void evaluate_unet::output(const EvaluateParam& param)
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
                    switch(input_size_strategy)
                    {
                    case 0: //match resolution
                        if(raw_image_vs[cur_output] != model->voxel_size)
                        {
                            tipl::resample_mt(from,to,
                                tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                                    raw_image_shape[cur_output],raw_image_vs[cur_output],
                                    model->dim,model->voxel_size));
                            return;
                        }
                    case 2: //original
                        tipl::draw(from,to);
                        return;
                    case 1: //match sizes
                    {
                        float target_vs = std::min({float(raw_image_shape[cur_output].width())*raw_image_vs[cur_output][0]/float(model->dim[0]),
                                                    float(raw_image_shape[cur_output].height())*raw_image_vs[cur_output][1]/float(model->dim[1]),
                                                    float(raw_image_shape[cur_output].depth())*raw_image_vs[cur_output][2]/float(model->dim[2])});
                        tipl::resample_mt(from,to,
                            tipl::transformation_matrix<float>(tipl::affine_transform<float>(),
                                raw_image_shape[cur_output],raw_image_vs[cur_output],
                                model->dim,tipl::vector<3>(target_vs,target_vs,target_vs)));
                        return;
                    }

                    }
                });

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

void evaluate_unet::start(const EvaluateParam& param)
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
    read_file(param);
    evaluate(param);
    output(param);
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
