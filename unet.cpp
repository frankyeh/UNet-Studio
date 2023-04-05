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
    std::vector<int> param({1,1});
    if(!mat.read("param",param) || !mat.read("feature_string",feature_string))
        return false;
    model = UNet3d(param[0],param[1],feature_string);
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
    mat.write("param",{model->in_count,model->out_count});
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
size_t get_cost(std::string feature_string)
{
    UNet3d model;
    model->feature_string = feature_string;
    return model->size();
}
std::vector<std::vector<int> > feature_string2features_vector(const std::string& feature_string)
{
    std::vector<std::vector<int> > features_vector;
    for(auto feature_string_per_level : tipl::split(feature_string,'+'))
    {
        features_vector.push_back(std::vector<int>());
        for(auto s : tipl::split(feature_string_per_level,'x'))
            features_vector.back().push_back(std::stoi(s));
    }
    return features_vector;
}

std::string features_vector2feature_string(const std::vector<std::vector<int> >& feature_vector)
{
    std::string result;
    for(int level = 0;level < feature_vector.size();++level)
    {
        if(level)
            result += "+";
        for(int b = 0;b < feature_vector[level].size();++b)
        {
            if(b)
                result += "x";
            result += std::to_string(feature_vector[level][b]);
        }
    }
    return result;
}

void gen_list(std::vector<std::string>& network_list)
{
    std::multimap<size_t,std::string> sorted_list;

    sorted_list.insert(std::make_pair(get_cost("8+8+8+8"),"8+8+8+8"));
    sorted_list.insert(std::make_pair(get_cost("8+8+8+8+8"),"8+8+8+8+8"));
    sorted_list.insert(std::make_pair(get_cost("8+8+8+8+8+8"),"8+8+8+8+8+8"));

    const size_t max_network = 1000;
    size_t count = 0;
    std::set<std::string> networks;
    networks.insert("8+8+8+8");
    networks.insert("8+8+8+8+8");
    networks.insert("8+8+8+8+8+8");
    for(auto beg = sorted_list.begin();beg != sorted_list.end() && count < max_network;++beg,++count)
    {
        auto cur_network = beg->second;
        network_list.push_back(cur_network);
        tipl::out() << beg->first << " " << cur_network << std::endl;
        auto feature_vector = feature_string2features_vector(cur_network);
        for(int level = 0;level < feature_vector.size();++level)
        {
            for(int b = 0;b < feature_vector[level].size();++b)
            {
                if((b == feature_vector[level].size()-1 && (level == feature_vector.size()-1 || feature_vector[level][b] != feature_vector[level+1][0])) ||
                   (b != feature_vector[level].size()-1 && feature_vector[level][b] != feature_vector[level][b+1]))
                {
                    feature_vector[level][b] *= 2;
                    auto new_network = features_vector2feature_string(feature_vector);
                    auto cost = get_cost(new_network);
                    if(networks.find(new_network) == networks.end())
                    {
                        sorted_list.insert(std::make_pair(cost,new_network));
                        networks.insert(new_network);
                    }
                    feature_vector[level][b] /= 2;
                    //tipl::out() << "adding " << new_network << " cost=" << cost << std::endl;
                }
            }
            if(feature_vector[level].size() < 3)
            if(level == feature_vector.size()-1 || feature_vector[level].size() < feature_vector[level+1].size())
            {
                feature_vector[level].push_back(feature_vector[level].back());
                auto new_network = features_vector2feature_string(feature_vector);
                auto cost = get_cost(new_network);
                if(networks.find(new_network) == networks.end())
                {
                    sorted_list.insert(std::make_pair(cost,new_network));
                    networks.insert(new_network);
                }
                feature_vector[level].pop_back();
                //tipl::out() << "adding " << new_network << " cost=" << cost << std::endl;
            }
        }
    }
}
