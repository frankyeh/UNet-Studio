#include "evaluate.hpp"

void evaluate_unet::read_file(const EvaluateParam& param)
{
    evaluate_image = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_result  = std::vector<tipl::image<3> >(param.image_file_name.size());
    evaluate_image_shape = std::vector<tipl::shape<3> >(param.image_file_name.size());
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
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);

                }
                tipl::vector<3> vs;
                auto& eval_I1 = evaluate_image[i];
                if(tipl::io::gz_nifti::load_from_file(param.image_file_name[i].c_str(),eval_I1,vs))
                {
                    tipl::affine_transform<float> arg;
                    tipl::vector<3> param_vs(1.0f,1.0f,1.0f);
                    if(vs[0] < 0.5f)
                        param_vs[2] = param_vs[1] = param_vs[0] = eval_I1.shape()[0]*vs[0]/float(param.dim[0]);
                    evaluate_image_trans[i] = tipl::transformation_matrix<float>(arg,param.dim,param_vs,eval_I1.shape(),vs);
                    evaluate_image_shape[i] = eval_I1.shape();

                    tipl::image<3> image_out(param.dim);
                    tipl::resample_mt(eval_I1,image_out,evaluate_image_trans[i]);
                    image_out.swap(eval_I1);
                    eval_I1 *= 2.0f/tipl::max_value(eval_I1);
                }
                else
                    eval_I1.resize(param.dim);
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
                    if(aborted)
                        return;
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(200ms);

                }
                auto& eval_I1 = evaluate_image[cur_prog];
                auto out = model->forward(torch::from_blob(&eval_I1[0],{1,model->in_count,int(eval_I1.depth()),int(eval_I1.height()),int(eval_I1.width())}).to(param.device));
                evaluate_result[cur_prog].resize(eval_I1.shape().multiply(tipl::shape<3>::z,model->out_count));
                evaluate_result[cur_prog] = 0;
                std::memcpy(&evaluate_result[cur_prog][0],out.to(torch::kCPU).data_ptr<float>(),evaluate_result[cur_prog].size()*sizeof(float));
            }
        }
        catch(...)
        {
            error_msg = "error occured during evaluation";
            std::cout << error_msg << std::endl;
            aborted = true;
        }
    }));
}

void evaluate_unet::output(void)
{
    evaluate_output = std::vector<tipl::image<3> >(evaluate_image.size());
    output_thread.reset(new std::thread([=](){
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
                evaluate_output[cur_output].resize(evaluate_image_shape[cur_output].multiply(tipl::shape<3>::z,model->out_count));
                evaluate_image_trans[cur_output].inverse();
                tipl::shape<3> dim_in(evaluate_image[cur_output].shape());
                tipl::shape<3> dim_out(evaluate_image_shape[cur_output]);
                tipl::par_for(model->out_count,[&](int i)
                {
                    auto from = evaluate_result[cur_output].sub_image(dim_in.size()*i,dim_in);
                    auto to = evaluate_output[cur_output].sub_image(dim_out.size()*i,dim_out);
                    tipl::resample_mt(from,to,evaluate_image_trans[cur_output]);
                });
                evaluate_image[cur_output] = tipl::image<3>();
                evaluate_result[cur_output] = tipl::image<3>();
            }
        }
        catch(...)
        {
            error_msg = "error occured during output";
            std::cout << error_msg << std::endl;
            aborted = true;
        }
        running = false;
    }));
}

void evaluate_unet::start(const EvaluateParam& param)
{
    stop();
    model->to(param.device);
    model->train();
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
