#ifndef UNET_HPP
#define UNET_HPP
#ifdef QT_CORE_LIB
    #undef slots
#endif
#include<torch/torch.h>
#ifdef QT_CORE_LIB
    #define slots Q_SLOTS
#endif
#include "TIPL/tipl.hpp"


struct UNet3dImpl : torch::nn::Module
{
public:
    int in_count = 1;
    int out_count = 1;
    std::string architecture,preproc,postproc,orientation,fov_strategy,error_msg;
public:
    std::shared_ptr<torch::optim::SGD> optimizer;

    std::vector<float> testing_errors,training_errors;
    std::vector<unsigned int> single_component_label;
    mutable std::mutex error_mutex;
    auto get_training_errors(void) const
    {
        std::scoped_lock<std::mutex> lock(error_mutex);
        return std::vector<float>(training_errors);
    }
    auto get_testing_errors(void) const
    {
        std::scoped_lock<std::mutex> lock(error_mutex);
        return std::vector<float>(testing_errors);
    }

public:
    tipl::vector<3> voxel_size = {1.0f,1.0f,1.0f};
    tipl::shape<3> dim = {192,224,192};
    std::deque<torch::nn::Sequential> encoding,decoding,decoding_tail;
    std::vector<torch::nn::Sequential> output;
    int create_layer(torch::nn::Sequential& layers,const std::string& def, int in_c);
public:
    std::string get_info(void) const;
public:
    UNet3dImpl(void){}
    UNet3dImpl(int32_t in_count_,int32_t out_count_,std::string);
    void copy_from(const UNet3dImpl& r);
    void add_gradient_from(const UNet3dImpl& r);
    void create_optimizer(float learning_rate);
public:
    std::vector<torch::Tensor> forward(torch::Tensor inputTensor);

    void set_requires_grad(bool req)
    {
        for (auto& p : parameters())
            p.set_requires_grad(req);
    }
    virtual void train(bool on = true) override
    {
        set_requires_grad(on);
        torch::nn::Module::train(on);
    }
    void print_layers(void);
    torch::Device device(void) const
    {
        return parameters().size() && parameters()[0].defined() ? parameters()[0].device() : torch::kCPU;
    }
    void prepare_for_inference(const torch::Device& device);
};
TORCH_MODULE_IMPL(UNet3d, UNet3dImpl);


#endif// UNET_HPP
