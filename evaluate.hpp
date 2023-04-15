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

class evaluate_unet{
public:
    std::vector<tipl::image<3> > evaluate_image,evaluate_result;
    std::vector<tipl::shape<3> > evaluate_image_shape;
    std::vector<tipl::vector<3> > evaluate_image_vs;
    std::vector<tipl::matrix<4,4> > evaluate_image_trans;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    void read_file(const EvaluateParam& param);
    void get_result(size_t index);
public:
    bool aborted = false;
    bool running = false;
    std::string error_msg;
private:
    size_t cur_prog = 0;
    std::shared_ptr<std::thread> evaluate_thread;
    void evaluate(const EvaluateParam& param);
private:
    std::shared_ptr<std::thread> output_thread;
    void output(void);
public:
    size_t cur_output = 0;
    std::vector<tipl::image<3> > evaluate_output;
    void clear(void)
    {
        stop();
        cur_output = 0;
        evaluate_output.clear();
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

