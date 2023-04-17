#include "evaluate.hpp"


tipl::shape<3> unet_inputsize(const tipl::shape<3>& s)
{
    return tipl::shape<3>(int(std::ceil(float(s[0])/32.0f))*32,int(std::ceil(float(s[1])/32.0f))*32,int(std::ceil(float(s[2])/32.0f))*32);
}
void evaluate_unet::read_file(const EvaluateParam& param)
{
    evaluate_image = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_result  = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
    evaluate_image_vs = std::vector<tipl::vector<3> >(param.image_file_name.size());
    evaluate_image_trans2mni = std::vector<tipl::matrix<4,4> >(param.image_file_name.size());
    evaluate_image_trans = std::vector<tipl::transformation_matrix<float> >(param.image_file_name.size());
    data_ready = std::vector<bool> (param.image_file_name.size());
    read_file_thread.reset(new std::thread([=]()
    {
        const int thread_count = std::thread::hardware_concurrency()/2;
        tipl::par_for(thread_count,[&](size_t thread)
        {
            for(size_t i = thread;i < evaluate_image.size() && !aborted;)
            {
                while(i > cur_prog+6)
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    if(aborted)
                        return;
                }
                if(tipl::io::gz_nifti::load_from_file(param.image_file_name[i].c_str(),evaluate_image[i],evaluate_image_vs[i],evaluate_image_trans2mni[i]))
                {
                    tipl::normalize(evaluate_image[i]);
                    auto inputsize = unet_inputsize(evaluate_image_shape[i] = evaluate_image[i].shape());

                    if(evaluate_image_vs[i] == model->voxel_size)
                    {
                        if(inputsize != evaluate_image_shape[i])
                        {
                            tipl::image<3> new_sized_image(inputsize);
                            tipl::draw(evaluate_image[i],new_sized_image,tipl::vector<3,int>(0,0,0));
                            new_sized_image.swap(evaluate_image[i]);
                        }
                    }
                    else
                    {
                        tipl::image<3> new_sized_image(inputsize);
                        evaluate_image_trans[i] =
                                tipl::transformation_matrix<float>(
                                    tipl::affine_transform<float>(),
                                    evaluate_image_shape[i],
                                    evaluate_image_vs[i],
                                    inputsize,model->voxel_size);
                        tipl::resample_mt(evaluate_image[i],new_sized_image,evaluate_image_trans[i]);
                        new_sized_image.swap(evaluate_image[i]);
                    }

                }
                data_ready[i] = true;
                i += thread_count;
            }
        });
    }));
}

void evaluate_unet::evaluate(const EvaluateParam& param)
{
    evaluate_thread.reset(new std::thread([=](){
        try{
            for (cur_prog = 0; cur_prog < evaluate_image.size() && !aborted; cur_prog++)
            {
                while(!data_ready[cur_prog])
                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);
                    if(aborted)
                        return;
                }
                auto& eval_I1 = evaluate_image[cur_prog];
                if(eval_I1.empty())
                    continue;
                auto out = model->forward(torch::from_blob(&eval_I1[0],{1,model->in_count,int(eval_I1.depth()),int(eval_I1.height()),int(eval_I1.width())}).to(param.device));
                evaluate_result[cur_prog].resize(eval_I1.shape().multiply(tipl::shape<3>::z,model->out_count));
                std::memcpy(&evaluate_result[cur_prog][0],out.to(torch::kCPU).data_ptr<float>(),evaluate_result[cur_prog].size()*sizeof(float));
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

void evaluate_unet::output(void)
{
    evaluate_output = std::vector<tipl::image<3> >(evaluate_image.size());
    is_label = std::vector<char>(evaluate_image.size());
    output_thread.reset(new std::thread([this](){
        try{
            for (cur_output = 0;cur_output < evaluate_image.size() && !aborted; cur_output++)
            {
                while(cur_output >= cur_prog)
                {
                    if(aborted)
                    {
                        running = false;
                        return;
                    }
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);

                }
                if(evaluate_result[cur_output].empty())
                    continue;
                evaluate_output[cur_output].resize(evaluate_image_shape[cur_output].multiply(tipl::shape<3>::z,model->out_count));
                tipl::shape<3> dim_in(evaluate_image[cur_output].shape());
                tipl::shape<3> dim_out(evaluate_image_shape[cur_output]);
                if(evaluate_image_vs[cur_output] == model->voxel_size)
                {
                    tipl::par_for(model->out_count,[&](int i)
                    {
                        auto from = evaluate_result[cur_output].alias(dim_in.size()*i,dim_in);
                        auto to = evaluate_output[cur_output].alias(dim_out.size()*i,dim_out);
                        tipl::draw(from,to,tipl::vector<3,int>(0,0,0));
                    });
                }
                else
                {
                    tipl::par_for(model->out_count,[&](int i)
                    {
                        auto from = evaluate_result[cur_output].alias(dim_in.size()*i,dim_in);
                        auto to = evaluate_output[cur_output].alias(dim_out.size()*i,dim_out);
                        auto T = evaluate_image_trans[cur_output];
                        T.inverse();
                        tipl::resample_mt(from,to,T);
                    });
                }
                evaluate_image[cur_output] = tipl::image<3>();
                evaluate_result[cur_output] = tipl::image<3>();
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
        running = false;
    }));
}

void evaluate_unet::start(const EvaluateParam& param)
{
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
