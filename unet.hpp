#ifndef UNET3D_HPP
#define UNET3D_HPP
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
    std::string feature_string;
    int kernel_size = 3;
public:
    uint32_t total_training_count = 0;
    tipl::vector<3> voxel_size = {1.0f,1.0f,1.0f};
    tipl::shape<3> dim = {192,224,192};
    std::deque<torch::nn::Sequential> encoding,decoding,up;
    std::vector<torch::nn::BatchNorm3d> bn_layers;
    torch::nn::Sequential output;
public:
    auto parse_feature_string(void) const
    {
        std::vector<std::vector<int> > features_down;
        std::vector<std::vector<int> > features_up;
        int input_feature = in_count;
        for(auto feature_string_per_level : tipl::split(feature_string,'+'))
        {
            features_down.push_back(std::vector<int>({input_feature}));
            for(auto s : tipl::split(feature_string_per_level,'x'))
                features_down.back().push_back(input_feature = std::stoi(s));
            features_up.push_back(std::vector<int>(features_down.back().rbegin(),features_down.back().rend()-1));
            features_up.back()[0] *= 2; // due to input concatenation
        }
        return std::make_pair(features_down,features_up);
    }
    std::string get_info(void) const
    {
        std::ostringstream out;
        out << "structure: " << feature_string << std::endl;
        out << "input: " << in_count << std::endl;
        out << "output: " << out_count << std::endl;
        out << "resolution: " << voxel_size << std::endl;
        out << "total training: " << total_training_count << std::endl;
        return out.str();
    }
public:
    UNet3dImpl(void){}
    UNet3dImpl(int32_t in_count_,
                int32_t out_count_,
                std::string feature_string_):
                in_count(in_count_),
                out_count(out_count_),
                feature_string(feature_string_)
    {
        auto features = parse_feature_string();
        std::vector<std::vector<int> > features_down(std::move(features.first));
        std::vector<std::vector<int> > features_up(std::move(features.second));

        for(int level=0; level< features_down.size(); level++)
        {
            encoding.push_back(
                ConvBlock(features_down[level],
                          level == 0 ? torch::nn::Sequential() : torch::nn::Sequential(torch::nn::MaxPool3d(torch::nn::MaxPool3dOptions(2).stride(2)))));
            register_module(std::string("encode")+std::to_string(level),encoding.back());
        }
        for(int level=features_down.size()-2; level>=0; level--)
        {
            up.push_front(
                ConvBlock({features_up[level+1].back(),features_down[level].back()},
                          torch::nn::Sequential(torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2, 2, 2})).mode(torch::kNearest)))));
            register_module("up"+std::to_string(level),up.front());
            decoding.push_front(
                ConvBlock(features_up[level]));
            register_module(std::string("decode")+std::to_string(level),decoding.front());
        }

        output=torch::nn::Sequential(
                   torch::nn::Conv3d(torch::nn::Conv3dOptions(features_up[0].back(), out_count, 1)));
        register_module("output",output);
    }
    void copy_from(UNet3dImpl& r)
    {
        set_requires_grad(false);
        auto rhs = r.parameters();
        auto lhs = parameters();
        for(size_t index = 0;index < rhs.size();++index)
        {
            if(lhs[index].sizes() == rhs[index].sizes())
                lhs[index].copy_(rhs[index].to(lhs[index].device()));
            else
            {
                if(rhs[index].numel() > lhs[index].numel())
                    lhs[index].copy_(rhs[index].reshape({rhs[index].numel()}).slice(0,0,lhs[index].numel()).reshape(lhs[index].sizes()).to(lhs[index].device()));
                else
                    lhs[index].reshape({lhs[index].numel()}).index_put_({torch::indexing::Slice(0,int(rhs[index].numel()))},
                                       rhs[index].to(lhs[index].device()).reshape({rhs[index].numel()}));
            }
        }

        total_training_count = r.total_training_count;
        voxel_size = r.voxel_size;
        dim = r.dim;
    }
    void set_requires_grad(bool req)
    {
        for (auto& p : parameters())
            p.set_requires_grad(req);
    }

    torch::Tensor forward(torch::Tensor inputTensor)
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
    void set_bn_tracking_running_stats(bool stat)
    {
        for(auto bn : bn_layers)
            bn->options.track_running_stats(stat);
    }
private:
    torch::nn::Sequential ConvBlock(const std::vector<int>& rhs,torch::nn::Sequential s = torch::nn::Sequential())
    {
        int count = 0;
        for(auto next_count : rhs)
        {
            if(count)
            {
                s->push_back(torch::nn::Conv3d(torch::nn::Conv3dOptions(count, next_count, kernel_size).padding((kernel_size-1)/2)));
                s->push_back(torch::nn::ReLU());
                auto bn = torch::nn::BatchNorm3d(next_count);
                s->push_back(bn);
                bn_layers.push_back(bn);
            }
            count = next_count;
        }
        return s;
    }
};
TORCH_MODULE_IMPL(UNet3d, UNet3dImpl);


#endif// UNET3D_HPP
