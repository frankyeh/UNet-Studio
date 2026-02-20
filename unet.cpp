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

        decoding.push_front(ConvBlock(features_up[level],ks[level]));
        register_module(std::string("decode")+std::to_string(level),decoding.front());

        output[level] = torch::nn::Sequential();
        output[level]->push_back("out_conv", torch::nn::Conv3d(torch::nn::Conv3dOptions(features_up[level].back(), out_count, 1)));
        register_module("output"+std::to_string(level), output[level]);
    }

    std::stringstream ss;
    // 1. Define Scope (Input -> Output)
    ss << "The implemented model is a 3D U-Net designed to map "
       << in_count << " input channel" << (in_count > 1 ? "s" : "") << " to "
       << out_count << " output classes using " << features_down.size() << " resolution levels. ";

    // 2. Define Architecture Depth
    ss << "The encoder pathway utilizes a feature hierarchy of [";
    for(size_t i = 0; i < features_down.size(); ++i)
        ss << (i > 0 ? ", " : "") << features_down[i].back();
    ss << "] channels. ";

    // 3. Define Mechanisms
    ss << "Downsampling is performed via 3D max pooling (stride 2), while the decoder employs "
       << "nearest-neighbor upsampling with skip connections via concatenation. ";

    // 4. Define Training Strategy
    ss << "To facilitate gradient flow, deep supervision is applied by generating auxiliary outputs "
       << "at the " << (features_down.size() - 1) << " upper resolution levels.";

    report = ss.str();
}

bool UNet3dImpl::init_dimension(const std::string& template_file)
{
    tipl::io::gz_nifti in(template_file,std::ios::in);
    if(!in)
    {
        error_msg = in.error_msg;
        return false;
    }
    in.toLPS();
    in >> std::tie(dim,voxel_size);
    dim = tipl::ml3d::round_up_size(dim);
    voxel_size = voxel_size[0];

    tipl::out() << "input dim: " << dim << " voxel size:" << voxel_size;
    std::stringstream ss;
    ss << " The input dimension was " << dim << ". The voxel size was " << voxel_size;
    report += ss.str();
    return true;
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
    int count = 0, idx = 0;
    for(auto next_count : rhs)
    {
        if(count)
        {
            std::string id = std::to_string(idx++);
            s->push_back("conv"+id, torch::nn::Conv3d(torch::nn::Conv3dOptions(count, next_count, ks).padding((ks-1)/2)));
            s->push_back("norm"+id, torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(next_count).affine(true)));
            s->push_back("relu"+id, torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)));
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
    voxel_size = r.voxel_size;
    dim = r.dim;
    report = r.report;
    errors = r.errors;
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
    out << "in: " << in_count << " out: " << out_count << std::endl;
    out << "dim: " << dim << " reso: " << voxel_size << std::endl;
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



