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
    std::string feature_string;
public:
    uint32_t total_training_count = 0;
    tipl::vector<3> voxel_size = {1.0f,1.0f,1.0f};
    tipl::shape<3> dim = {192,224,192};
    std::deque<torch::nn::Sequential> encoding,decoding,up;
    std::vector<torch::nn::BatchNorm3d> bn_layers;
    torch::nn::Sequential output;
public:
    auto parse_feature_string(std::vector<int>& kernel_size) const
    {
        std::vector<std::vector<int> > features_down;
        std::vector<std::vector<int> > features_up;
        int input_feature = in_count;
        for(auto feature_string_per_level : tipl::split(feature_string,'+'))
        {
            auto level_feature_string = tipl::split(feature_string_per_level,',');
            if(level_feature_string.size() == 2)
                kernel_size.push_back(std::stoi(level_feature_string.back()));
            else
                kernel_size.push_back(3);
            features_down.push_back(std::vector<int>({input_feature}));
            for(auto s : tipl::split(feature_string_per_level,'x'))
                features_down.back().push_back(input_feature = std::stoi(s));
            features_up.push_back(std::vector<int>(features_down.back().rbegin(),features_down.back().rend()-1));
            features_up.back()[0] *= 2; // due to input concatenation
        }
        return std::make_pair(features_down,features_up);
    }
    std::string get_info(void) const;
public:
    UNet3dImpl(void){}
    UNet3dImpl(int32_t in_count_,
                int32_t out_count_,
                std::string feature_string_);
    void copy_from(const UNet3dImpl& r);

public:
    torch::Tensor forward(torch::Tensor inputTensor);

    void set_requires_grad(bool req)
    {
        for (auto& p : parameters())
            p.set_requires_grad(req);
    }

    void set_bn_tracking_running_stats(bool stat)
    {
        for(auto bn : bn_layers)
            bn->options.track_running_stats(stat);
    }
    virtual void train(bool on = true) override
    {
        set_requires_grad(on);
        set_bn_tracking_running_stats(on);
        torch::nn::Module::train(on);
    }
    void print_layers(void);
private:
    torch::nn::Sequential ConvBlock(const std::vector<int>& rhs,size_t ks,torch::nn::Sequential s = torch::nn::Sequential());
};
TORCH_MODULE_IMPL(UNet3d, UNet3dImpl);


#endif// UNET_HPP
