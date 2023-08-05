#ifndef EVALUATE_HPP
#define EVALUATE_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"

struct EvaluateParam{
    std::vector<std::string> image_file_name;
    torch::Device device = torch::kCPU;
    float prob_threshold = 0.5f;
};
struct ProcStrategy{
    // preproc
    bool match_resolution = true;
    bool match_fov = true;
    bool match_orientation = false;
    std::string template_file_name;
    // postproc
    unsigned char output_format = 0;
};

class OptionTableWidget;
class evaluate_unet{
public:
    OptionTableWidget* option = nullptr;
    EvaluateParam param;
public:
    ProcStrategy proc_strategy;
    std::vector<tipl::image<3> > network_input,network_output;
    std::vector<tipl::shape<3> > raw_image_shape;
    std::vector<tipl::vector<3> > raw_image_vs;
    std::vector<tipl::transformation_matrix<float> > raw_image_trans;
    std::vector<std::vector<char> > raw_image_flip_swap;
    std::vector<tipl::image<3,char> > raw_image_mask;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    void read_file(void);
    void get_result(size_t index);
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
    void stop(void);
    bool save_to_file(size_t index,const char* file_name);

};

#endif // EVALUATE_HPP

