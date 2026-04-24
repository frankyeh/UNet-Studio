#ifndef EVALUATE_HPP
#define EVALUATE_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"
#include "TIPL/tipl.hpp"

struct EvaluateParam{
    std::vector<std::string> image_file_name;
    torch::Device device = torch::kCPU;
    float prob_threshold = 0.5f;
};
struct ProcStrategy{
    // preproc
    unsigned char output_format = 0;
};

class evaluate_unet{
public:
    EvaluateParam param;
public:
    ProcStrategy proc_strategy;
    std::vector<tipl::ml3d::evalution_set<tipl::image<3>>> eval;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    void read_file(void);
    void get_result(size_t index);
public:
    std::vector<std::string> tissue_names = {"background","white matter","gray matter","cerebellar gray matter","subcortical"};
    tipl::image<3,unsigned char> template_I;
    tipl::image<3,unsigned short> atlas_I;
    tipl::matrix<4,4,float> template_R;
    tipl::vector<3> template_vs;
    size_t atlas_region_count = 0;
    bool load_template(const std::string& file_name)
    {
        if(!(tipl::io::gz_nifti(file_name,std::ios::in) >> template_I >> template_R >> template_vs
                >> [&](const std::string& e){error_msg = e;}))
            return false;
        // remove csf region
        std::replace_if(template_I.begin(),template_I.end(),[](auto v){return v >= 5;},0);
        return true;
    }
    bool load_atlas(const std::string& file_name);
public:
    bool aborted = false;
    bool running = false;
    std::string status,error_msg;
private:
    size_t cur_prog = 0;
    std::shared_ptr<std::thread> evaluate_thread;
    void evaluate(void);
private:
    std::shared_ptr<std::thread> output_thread;
    void output(void);
public:
    size_t cur_output = 0;
    std::vector<tipl::image<3> > label_prob,foreground_prob;
    void proc_actions(const char* cmd,float param1 = 0.0f,float param2 = 0.0f);
    std::vector<char> is_label;
    void clear(void)
    {
        stop();
        cur_output = 0;
        label_prob.clear();
    }
public:
    UNet3d model;
public:
    ~evaluate_unet(void)
    {
        stop();
    }
    void start(void);
    void join(void);
    void stop(void);
    bool save_to_file(size_t index,const char* file_name);

};

#endif // EVALUATE_HPP

