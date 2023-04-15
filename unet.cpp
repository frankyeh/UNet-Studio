#include <map>
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"


bool load_from_file(UNet3d& model,const char* file_name)
{
    tipl::io::gz_mat_read mat;
    if(!mat.load_from_file(file_name))
        return false;
    std::string feature_string;
    std::vector<int> param({1,1,0});
    if(!mat.read("param",param) || !mat.read("feature_string",feature_string))
        return false;
    model = UNet3d(param[0],param[1],feature_string);
    model->total_training_count = param[2];
    model->train();
    int id = 0;
    for(auto& tensor : model->parameters())
    {
        unsigned int row,col;
        const auto* data = mat.read_as_type<float>((std::string("tensor")+std::to_string(id)).c_str(),row,col);
        if(!data || row*col != tensor.numel())
            return false;
        std::copy(data,data+row*col,tensor.data_ptr<float>());
        ++id;
    }
    return true;
}
bool save_to_file(UNet3d& model,const char* file_name)
{
    tipl::io::gz_mat_write mat(file_name);
    if(!mat)
        return false;
    mat.write("feature_string",model->feature_string);
    mat.write("param",{model->in_count,model->out_count,int(model->total_training_count)});
    int id = 0;
    for(auto tensor : model->parameters())
    {
        auto cpu_tensor = tensor.to(torch::kCPU);
        mat.write((std::string("tensor")+std::to_string(id)).c_str(),cpu_tensor.data_ptr<float>(),cpu_tensor.numel(),1);
        ++id;
    }
    return true;
}
std::string show_structure(const UNet3d& model)
{
    std::ostringstream out;
    auto features = model->parse_feature_string();
    std::vector<std::vector<int> > features_down(std::move(features.first));
    std::vector<std::vector<int> > features_up(std::move(features.second));

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

