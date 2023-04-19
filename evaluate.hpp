#ifndef EVALUATE_HPP
#define EVALUATE_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"

struct EvaluateParam{
    std::vector<std::string> image_file_name;
    torch::Device device = torch::kCPU;
};
class OptionTableWidget;
class evaluate_unet{
public:
    OptionTableWidget* option = nullptr;
public:
    unsigned char input_size_strategy = 0; //
    std::vector<tipl::image<3> > network_input,network_output;
    std::vector<tipl::shape<3> > raw_image_shape;
    std::vector<tipl::vector<3> > raw_image_vs;
    std::vector<tipl::matrix<4,4> > raw_image_trans2mni;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    void read_file(const EvaluateParam& param);
    void get_result(size_t index);
public:
    bool aborted = false;
    bool running = false;
    std::string status,error_msg;
private:
    size_t cur_prog = 0;
    std::shared_ptr<std::thread> evaluate_thread;
    void evaluate(const EvaluateParam& param);
private:
    std::shared_ptr<std::thread> output_thread;
    void output(const EvaluateParam& param);
public:
    size_t cur_output = 0;
    std::vector<tipl::image<3> > label_prob;
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
    void start(const EvaluateParam& param);
    void stop(void);

};

#endif // EVALUATE_HPP

