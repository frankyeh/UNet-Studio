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
    }

    output=torch::nn::Sequential(
               torch::nn::Conv3d(torch::nn::Conv3dOptions(features_up[0].back(), out_count, 1)));
    register_module("output",output);
}

torch::Tensor UNet3dImpl::forward(torch::Tensor inputTensor)
{
    std::vector<torch::Tensor> encodingTensor;
    for(int level=0; level< encoding.size(); level++)
        encodingTensor.push_back(inputTensor = encoding[level]->forward(inputTensor));

    for(int level=encoding.size()-2; level>=0; level--)
    {
        encodingTensor.pop_back();
        inputTensor=decoding[level]->forward(torch::cat({encodingTensor.back(),up[level]->forward(inputTensor)},1));
    }

    return output->forward(inputTensor);
}

torch::nn::Sequential UNet3dImpl::ConvBlock(const std::vector<int>& rhs,size_t ks,torch::nn::Sequential s)
{
    int count = 0;
    for(auto next_count : rhs)
    {
        if(count)
        {
            s->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(count, next_count, ks).padding((ks-1)/2)));
            s->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)));
            auto bn = torch::nn::BatchNorm3d(next_count);
            s->push_back(bn);
            bn_layers.push_back(bn);
        }
        count = next_count;
    }
    return s;
}

void UNet3dImpl::copy_from(const UNet3dImpl& r)
{
    auto rhs = r.parameters();
    auto lhs = parameters();
    tipl::adaptive_par_for(rhs.size(),[&](size_t index)
    {
        torch::NoGradGuard no_grad;
        bool requires_grad = lhs[index].requires_grad();
        lhs[index].set_requires_grad(false);
        auto new_rhs = rhs[index].to(lhs[index].device()).detach();
        if(lhs[index].sizes() == rhs[index].sizes())
        {
            lhs[index].copy_(new_rhs);
            if(lhs[index].mutable_grad().defined() && rhs[index].mutable_grad().defined())
                lhs[index].mutable_grad().copy_(rhs[index].mutable_grad().to(lhs[index].device()).detach());
        }
        else
        {
            if(rhs[index].numel() > lhs[index].numel())
                lhs[index].copy_(new_rhs.reshape({rhs[index].numel()}).slice(0,0,lhs[index].numel()).reshape(lhs[index].sizes()));
            else
                lhs[index].reshape({lhs[index].numel()}).index_put_({torch::indexing::Slice(0,int(rhs[index].numel()))},
                                   new_rhs.reshape({rhs[index].numel()}));
        }
        lhs[index].set_requires_grad(requires_grad);
    });

    auto rhs_buffers = r.buffers();
    auto lhs_buffers = buffers();
    auto rhs_iter = rhs_buffers.begin();
    auto lhs_iter = lhs_buffers.begin();
    while (rhs_iter != rhs_buffers.end() && lhs_iter != lhs_buffers.end())
    {
        const auto& rhs_tensor = *rhs_iter;
        auto& lhs_tensor = *lhs_iter;

        if (lhs_tensor.sizes() == rhs_tensor.sizes())
            lhs_tensor.copy_(rhs_tensor.to(lhs_tensor.device()).detach());
        else
        {
            // Handle size mismatch
            auto reshaped_rhs = rhs_tensor.to(lhs_tensor.device()).detach().reshape(lhs_tensor.sizes());
            lhs_tensor.copy_(reshaped_rhs);
        }
        ++rhs_iter;
        ++lhs_iter;
    }


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



