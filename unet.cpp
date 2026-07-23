#include <map>
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"

void UNet3dImpl::prepare_for_inference(const torch::Device& device)
{
    to(device);
    eval();

    for(auto& module : modules())
        if(auto bn =
            std::dynamic_pointer_cast<torch::nn::BatchNorm3dImpl>(module))
        {
            bn->eval();
            bn->running_mean.zero_();
            bn->running_var.fill_(1.0f);
            if(bn->num_batches_tracked.defined())
                bn->num_batches_tracked.zero_();
        }
}

int UNet3dImpl::create_layer(torch::nn::Sequential& layers,const std::string& def,int in_c)
{
    std::unordered_map<std::string,std::string> params;
    for(const auto& arg : tipl::split(def,','))
    {
        size_t pos = arg.find_first_of("0123456789");
        if(pos != std::string::npos)
            params[arg.substr(0,pos)] = arg.substr(pos);
        else
            params[arg] = "1";
    }

    int out_c = in_c;

    if(params.count("max_pool"))
        layers->push_back(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).stride(2)));
    else
    if(params.count("upsample"))
        layers->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
            .scale_factor(std::vector<double>{2.0,2.0,2.0})
            .mode(torch::kNearest)));
    else
    if(params.count("conv_trans"))
    {
        out_c = std::stoi(params["conv_trans"]);
        int ks = params.count("ks") ? std::stoi(params["ks"]) : 2;
        int stride = params.count("stride") ? std::stoi(params["stride"]) : 2;

        if(ks != 2 || stride != 2)
            throw std::runtime_error("conv_trans supports only ks2 stride2");

        layers->push_back(torch::nn::ConvTranspose3d(
            torch::nn::ConvTranspose3dOptions(in_c,out_c,ks).stride(stride)));
    }
    else
    if(params.count("conv"))
    {
        out_c = std::stoi(params["conv"]);
        int ks = params.count("ks") ? std::stoi(params["ks"]) : 3;
        int stride = params.count("stride") ? std::stoi(params["stride"]) : 1;

        if(!((ks == 1 && stride == 1) || (ks == 3 && (stride == 1 || stride == 2))))
            throw std::runtime_error("conv supports only ks1 stride1, ks3 stride1, and ks3 stride2");

        layers->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(in_c,out_c,ks)
            .stride(stride)
            .padding((ks-1)/2)));

    }
    else
    if(params.count("norm"))
    {
        layers->push_back(torch::nn::InstanceNorm3d(
            torch::nn::InstanceNorm3dOptions(in_c).affine(true)));
    }
    else
    if(params.count("bnorm"))
    {
        layers->push_back(torch::nn::BatchNorm3d(
            torch::nn::BatchNorm3dOptions(in_c).affine(true).track_running_stats(true).eps(0.0)));
    }
    else
    {
        std::string unknown_layer = params.empty() ? def : params.begin()->first;
        throw std::runtime_error("unknown layer: " + unknown_layer);
    }

    if(params.count("relu"))
        layers->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    else
    if(params.count("leaky_relu"))
        layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.01).inplace(true)));
    else
    if(params.count("elu"))
        layers->push_back(torch::nn::ELU(torch::nn::ELUOptions().inplace(true)));

    return out_c;
}

UNet3dImpl::UNet3dImpl(int32_t in_count_,
            int32_t out_count_,
            std::string architecture_):
            in_count(in_count_),
            out_count(out_count_),
            architecture(architecture_)
{
    fov_strategy = "align_top";
    preproc = "";
    postproc = "softmax+create_mask+argmax";
    std::vector<std::vector<std::string>> enc_tokens, dec_tokens;
    {
        std::vector<std::string> all_lines(tipl::split_by_line_breaks(architecture_));
        if(all_lines.size() < 3)
            throw std::runtime_error("invalid u-net structure");
        size_t enc_count = all_lines.size() / 2 + 1;
        for(size_t i = 0;i < all_lines.size();++i)
            (i < enc_count ? enc_tokens : dec_tokens).push_back(tipl::split(all_lines[i],'+'));
    }

    encoding.resize(enc_tokens.size());
    size_t channel = in_count_;
    std::vector<size_t> skip_channels(enc_tokens.size());
    for(int level = 0;level < enc_tokens.size();++level)
    {
        for(const auto& token : enc_tokens[level])
            channel = create_layer(encoding[level],token,channel);
        register_module(std::string("encode")+std::to_string(level), encoding[level]);
        skip_channels[level] = channel;
    }

    decoding.resize(dec_tokens.size());
    output.resize(dec_tokens.size());
    decoding_tail.resize(dec_tokens.size());

    auto out_token = dec_tokens.back().back();
    for(int level = dec_tokens.size()-1;level >= 0;--level)
    {
        const auto& tokens = dec_tokens[dec_tokens.size()-1-level];

        bool after_out = false;
        channel += skip_channels[level];

        for(size_t t = 0;t < tokens.size();++t)
        {
            if(tokens[t] == out_token)
            {
                create_layer(output[level],tokens[t],channel);
                after_out = true;
                continue;
            }

            channel = create_layer(after_out ? decoding_tail[level] : decoding[level],
                                   tokens[t],
                                   channel);
        }

        register_module("decode"+std::to_string(level),decoding[level]);
        if(!output[level]->is_empty())
            register_module("output"+std::to_string(level),output[level]);
        if(!decoding_tail[level]->is_empty())
            register_module("decode_tail"+std::to_string(level),decoding_tail[level]);
    }
}

std::vector<torch::Tensor> UNet3dImpl::forward(torch::Tensor inputTensor)
{
    std::vector<torch::Tensor> encodingTensors(encoding.size() - 1);
    std::vector<torch::Tensor> results(output.size());

    for(int level=0; level < encoding.size(); level++)
    {
        inputTensor = encoding[level]->forward(inputTensor);
        if (level < encoding.size() - 1)
            encodingTensors[level] = inputTensor;
    }
    for(int level = int(encoding.size())-2;level >= 0;--level)
    {
        inputTensor = torch::cat({encodingTensors[level],inputTensor},1);
        encodingTensors[level] = torch::Tensor();

        inputTensor = decoding[level]->forward(inputTensor);

        if(!output[level]->is_empty())
            results[level] = output[level]->forward(inputTensor);
        if(!decoding_tail[level]->is_empty())
            inputTensor = decoding_tail[level]->forward(inputTensor);
    }

    return results;
}

void UNet3dImpl::copy_from(const UNet3dImpl& r)
{
    auto rhs_params = r.parameters();
    auto lhs_params = parameters();

    for(size_t i=0; i<rhs_params.size(); ++i)
    {
        torch::NoGradGuard no_grad;
        if(lhs_params[i].sizes()==rhs_params[i].sizes())
            lhs_params[i].copy_(rhs_params[i]);
    }

    auto rhs_buffers = r.buffers();
    auto lhs_buffers = buffers();

    for(size_t i=0; i<rhs_buffers.size()&&i<lhs_buffers.size(); ++i)
    {
        torch::NoGradGuard no_grad;
        if(lhs_buffers[i].sizes()==rhs_buffers[i].sizes())
            lhs_buffers[i].copy_(rhs_buffers[i]);
    }

    voxel_size = r.voxel_size;
    dim = r.dim;
    fov_strategy = r.fov_strategy;
    postproc = r.postproc;
    preproc = r.preproc;
}

void UNet3dImpl::add_gradient_from(const UNet3dImpl& r)
{
    auto rhs = r.parameters();
    auto lhs = parameters();
    auto cur_device = device();

    tipl::par_for(rhs.size(), [&](size_t index)
    {
        torch::NoGradGuard no_grad;

        if (rhs[index].mutable_grad().defined())
        {
            auto rhs_grad = rhs[index].mutable_grad().to(cur_device).to(torch::kFloat32).detach();

            if (lhs[index].mutable_grad().defined())
                lhs[index].mutable_grad().add_(rhs_grad);
            else
                lhs[index].mutable_grad() = rhs_grad.clone();
        }
    });
}

void UNet3dImpl::create_optimizer(float learning_rate)
{
    tipl::progress prog("initialize optimizer");
    std::vector<torch::Tensor> decay_params,no_decay_params;
    for(auto& p : named_parameters())
    {
        auto v = p.value();
        const auto& name = p.key();
        bool no_decay = name.find("bias") != std::string::npos || v.dim() <= 1; // norm affine weights and all bias-like parameters
        if(no_decay)
            no_decay_params.push_back(v);
        else
            decay_params.push_back(v);
    }

    double base_wd = 3e-5;
    std::vector<torch::optim::OptimizerParamGroup> groups;

    auto opt_d = std::make_unique<torch::optim::SGDOptions>(learning_rate);
    opt_d->momentum(0.99);
    opt_d->nesterov(true);
    opt_d->weight_decay(base_wd);

    auto opt_nd = std::make_unique<torch::optim::SGDOptions>(learning_rate);
    opt_nd->momentum(0.99);
    opt_nd->nesterov(true);
    opt_nd->weight_decay(0.0);

    groups.push_back(torch::optim::OptimizerParamGroup(decay_params,std::move(opt_d)));
    groups.push_back(torch::optim::OptimizerParamGroup(no_decay_params,std::move(opt_nd)));
    optimizer = std::make_shared<torch::optim::SGD>(groups,torch::optim::SGDOptions(learning_rate));
}

std::string UNet3dImpl::get_info(void) const
{
    std::ostringstream out;
    out << "in: " << in_count << " out: " << out_count << std::endl;
    out << "dim: " << dim << " reso: " << voxel_size << std::endl;
    out << "structure: " << architecture << std::endl;
    if(!preproc.empty())
        out << "preproc: " << preproc << std::endl;
    if(!postproc.empty())
        out << "postproc: " << postproc << std::endl;
    return out.str();
}

void UNet3dImpl::print_layers(void)
{
    for(auto& module:modules())
    {
        if(!module->modules(false).empty())
            continue;
        std::cout << module->name();
        for(auto& tensor:module->parameters(true))
            std::cout << " " << tensor.sizes();
        std::cout << std::endl;
    }
}

