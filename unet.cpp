#include <map>
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"


UNet3dImpl::UNet3dImpl(int32_t in_count_,
            int32_t out_count_,
            std::string feature_string_):
            in_count(in_count_),
            out_count(out_count_),
            feature_string(feature_string_)
{
    std::vector<int> ks;
    auto features = parse_feature_string(ks);
    std::vector<std::vector<int> > features_down(std::move(features.first));
    std::vector<std::vector<int> > features_up(std::move(features.second));
    output.resize(features_down.size() - 1);

    for(int level=0; level< features_down.size(); level++)
    {
        encoding.push_back(
            ConvBlock(features_down[level],ks[level],
                      level == 0 ? torch::nn::Sequential() : torch::nn::Sequential(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).stride(2)))));
        register_module(std::string("encode")+std::to_string(level),encoding.back());
    }
    for(int level=features_down.size()-2; level>=0; level--)
    {
        up.push_front(
            ConvBlock({features_up[level+1].back(),features_down[level].back()},ks[level],
                      torch::nn::Sequential(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2, 2, 2})).mode(torch::kNearest)))));
        register_module("up"+std::to_string(level),up.front());
        decoding.push_front(
            ConvBlock(features_up[level],ks[level]));
        register_module(std::string("decode")+std::to_string(level),decoding.front());

        output[level] = torch::nn::Sequential(torch::nn::Conv3d(torch::nn::Conv3dOptions(features_up[level].back(), out_count, 1)));
        register_module("output"+std::to_string(level), output[level]);
    }

}

std::vector<torch::Tensor> UNet3dImpl::forward(torch::Tensor inputTensor)
{
    std::vector<torch::Tensor> encodingTensors;
    std::vector<torch::Tensor> results(output.size());

    // 1. Encoder Path
    for(int level=0; level < encoding.size(); level++) {
        inputTensor = encoding[level]->forward(inputTensor);
        encodingTensors.push_back(inputTensor);
    }

    // 2. Decoder Path
    // inputTensor currently holds the bottleneck (the last element in encodingTensors)
    for(int level = encoding.size() - 2; level >= 0; level--)
    {
        // Use the index 'level' to get the skip connection from the encoder
        auto x1 = encodingTensors[level];

        // Upsample the current features
        auto x2 = up[level]->forward(inputTensor);

        // Handle Odd Input Sizes (Padding)
        int diffD = x1.size(2) - x2.size(2);
        int diffH = x1.size(3) - x2.size(3);
        int diffW = x1.size(4) - x2.size(4);

        if(diffD > 0 || diffH > 0 || diffW > 0)
            x2 = torch::nn::functional::pad(x2, torch::nn::functional::PadFuncOptions({0, diffW, 0, diffH, 0, diffD}));

        // Concatenate encoder features (x1) and upsampled decoder features (x2)
        inputTensor = decoding[level]->forward(torch::cat({x1, x2}, 1));

        // Deep Supervision: level 0 is full res, level 1 is 1/2 res...
        results[level] = output[level]->forward(inputTensor);
    }
    return results;
}

torch::nn::Sequential UNet3dImpl::ConvBlock(const std::vector<int>& rhs,size_t ks,torch::nn::Sequential s)
{
    int count = 0;
    for(auto next_count : rhs)
    {
        if(count)
        {
            s->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(count, next_count, ks).padding((ks-1)/2)));
            s->push_back(torch::nn::BatchNorm3d(next_count));
            s->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)));
        }
        count = next_count;
    }
    return s;
}

void UNet3dImpl::copy_from(const UNet3dImpl& r)
{
    // 1. Copy Parameters (Weights, Biases)
    auto rhs_params = r.parameters();
    auto lhs_params = parameters();

    for(size_t i = 0; i < rhs_params.size(); ++i)
    {
        torch::NoGradGuard no_grad;
        if(lhs_params[i].sizes() == rhs_params[i].sizes())
            lhs_params[i].copy_(rhs_params[i]);
    }

    // 2. [CRITICAL] Copy Buffers (BN Running Mean, BN Running Var)
    auto rhs_buffers = r.buffers();
    auto lhs_buffers = buffers();

    // Ensure you are using the same number of buffers
    for(size_t i = 0; i < rhs_buffers.size() && i < lhs_buffers.size(); ++i)
    {
        torch::NoGradGuard no_grad;
        if(lhs_buffers[i].sizes() == rhs_buffers[i].sizes())
            lhs_buffers[i].copy_(rhs_buffers[i]);
    }

    // 3. Copy metadata
    total_training_count = r.total_training_count;
    voxel_size = r.voxel_size;
    dim = r.dim;
}
void UNet3dImpl::add_gradient_from(const UNet3dImpl& r)
{
    auto rhs = r.parameters();
    auto lhs = parameters();
    auto cur_device = device();
    tipl::adaptive_par_for(rhs.size(),[&](size_t index)
    {
        torch::NoGradGuard no_grad;
        if(lhs[index].mutable_grad().defined() && rhs[index].mutable_grad().defined())
            lhs[index].mutable_grad().add_(rhs[index].mutable_grad().to(cur_device).detach());
    });
}


std::string UNet3dImpl::get_info(void) const
{
    std::ostringstream out;
    out << "structure: " << feature_string << std::endl;
    out << "input: " << in_count << std::endl;
    out << "output: " << out_count << std::endl;
    out << "input/output sizes: " << dim << std::endl;
    out << "resolution: " << voxel_size << std::endl;
    out << "total training: " << total_training_count << std::endl;
    return out.str();
}
void UNet3dImpl::print_layers(void)
{
    for(auto& module : modules())
    {
        if(!module->modules(false).empty())
            continue;
        std::cout << module->name();
        for(auto& tensor : module->parameters(true))
            std::cout << " " << tensor.sizes();
        std::cout << std::endl;
    }
}



