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
    std::vector<std::vector<int>> features_down,features_up;
    tipl::ml3d::parse_feature_string(feature_string,in_count,features_down,features_up,ks);
    output.resize(features_down.size()-1);


    auto ConvBlock = [&](const std::vector<int>& rhs, size_t ks, int first_stride)
    {
        torch::nn::Sequential s;
        int count = 0, idx = 0;
        for(auto next_count:rhs)
        {
            if(count)
            {
                std::string id = std::to_string(idx++);
                int current_stride = (idx == 1) ? first_stride : 1;

                s->push_back("conv"+id, torch::nn::Conv3d(torch::nn::Conv3dOptions(count, next_count, ks).stride(current_stride).padding((ks-1)/2)));
                s->push_back("norm"+id, torch::nn::InstanceNorm3d(torch::nn::InstanceNorm3dOptions(next_count).affine(true)));
                s->push_back("relu"+id, torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().inplace(true)));
            }
            count = next_count;
        }
        return s;
    };

    for(int level=0; level<features_down.size(); level++)
    {
        int first_stride = (level == 0) ? 1 : 2;
        encoding.push_back(ConvBlock(features_down[level], ks[level], first_stride));
        register_module(std::string("encode")+std::to_string(level), encoding.back());
    }

    for(int level=features_down.size()-2; level>=0; level--)
    {
        up.push_front(torch::nn::Sequential(torch::nn::ConvTranspose3d(
                    torch::nn::ConvTranspose3dOptions(features_up[level+1].back(),features_down[level].back(),2).stride(2)
                )));
        register_module("up"+std::to_string(level),up.front());

        std::vector<int> current_decoder_features = features_up[level];

        // nnU-Net STRICT REQUIREMENT: Every decoder block must have exactly 2 convolutions.
        // If the parsed feature string only gives us [Concat_Channels, Out_Channels],
        // we add the Out_Channels again to make it [Concat_Channels, Out_Channels, Out_Channels].
        if(current_decoder_features.size() == 2)
            current_decoder_features.push_back(current_decoder_features.back());

        decoding.push_front(ConvBlock(current_decoder_features, ks[level], 1));
        register_module(std::string("decode")+std::to_string(level), decoding.front());

        output[level] = torch::nn::Sequential();
        output[level]->push_back("out_conv",torch::nn::Conv3d(
                    torch::nn::Conv3dOptions(features_up[level].back(),out_count,1)
                ));
        register_module("output"+std::to_string(level),output[level]);
    }

    std::stringstream ss;
    ss << "The implemented model is a 3D U-Net designed to map "
       << in_count << " input channel" << (in_count>1?"s":"") << " to "
       << out_count << " output classes using " << features_down.size() << " resolution levels. ";

    ss << "The encoder pathway utilizes a feature hierarchy of [";
    for(size_t i=0; i<features_down.size(); ++i)
        ss << (i>0?", ":"") << features_down[i].back();
    ss << "] channels. ";

    ss << "Downsampling is performed via 3D strided convolutions (stride 2), while the decoder employs "
       << "transposed convolutions for upsampling with skip connections via concatenation. ";

    ss << "To facilitate gradient flow, deep supervision is applied by generating auxiliary outputs "
       << "at the " << (features_down.size()-1) << " upper resolution levels.";

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
    std::vector<torch::Tensor> encodingTensors(encoding.size() - 1);
    std::vector<torch::Tensor> results(output.size());

    for(int level=0; level<encoding.size(); level++)
    {
        inputTensor = encoding[level]->forward(inputTensor);
        if (level < encoding.size() - 1)
            encodingTensors[level] = inputTensor;

    }

    for(int level=encoding.size()-2; level>=0; level--)
    {
        torch::Tensor x2 = up[level]->forward(inputTensor);
        inputTensor = torch::Tensor();
        torch::Tensor cat_tensor = torch::cat({encodingTensors[level],x2}, 1);
        x2 = torch::Tensor();
        encodingTensors[level] = torch::Tensor();
        inputTensor = decoding[level]->forward(cat_tensor);
        cat_tensor = torch::Tensor();
        results[level] = output[level]->forward(inputTensor);
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
    report = r.report;
    errors = r.errors;
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

std::string show_structure(const UNet3d& model)
{
    std::ostringstream out;
    std::vector<int> ks;
    std::vector<std::vector<int> > features_down,features_up;
    tipl::ml3d::parse_feature_string(model->feature_string,model->in_count,features_down,features_up,ks);

    for(int level=0; level< features_down.size(); level++)
    {
        for(auto i : features_down[level])
            out << std::string(level,'\t') << i << std::endl;
    }
    for(int level=features_down.size()-2; level>=0; level--)
    {
        out << std::string(level,'\t') << features_down[level].back() << "+" << features_down[level].back() << "<-" << features_up[level+1].back() << std::endl;
        for(auto i : features_up[level])
            out << std::string(level,'\t') << i << std::endl;
    }
    return out.str();
}
