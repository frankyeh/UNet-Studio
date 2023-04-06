#ifndef TRAIN_HPP
#define TRAIN_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"

struct TrainParam{
    std::vector<std::string> image_file_name;
    std::vector<std::string> label_file_name;
    int batch_size = 6;
    int epoch = 1000;
    bool from_scratch = true;
    float learning_rate = 0.01f;
    tipl::shape<3> dim;
    torch::Device device = torch::kCPU;
};
bool save_to_file(UNet3d& model,const char* file_name);
bool load_from_file(UNet3d& model,const char* file_name);
std::string show_structure(const UNet3d& model);
bool get_label_info(const std::string& label_name,int& out_count);
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          tipl::vector<3>& vs);
void load_image_and_label(tipl::image<3>& image,
                          tipl::image<3>& label,
                          const tipl::vector<3>& image_vs,
                          const tipl::shape<3>& template_shape,
                          size_t random_seed);




class train_unet{
public:
    bool aborted = false;
    bool pause = false;
    bool running = false;
    std::string error_msg;
private:
    std::vector<tipl::image<3> > in_data,out_data;
    std::vector<std::vector<float> > out_data_weight;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    void read_file(const TrainParam& param);
private:
    std::vector<torch::Tensor> in_tensor,out_tensor,out_tensor_weight;
    std::vector<bool> tensor_ready;
    std::shared_ptr<std::thread> prepare_tensor_thread;
    void prepare_tensor(const TrainParam& param);
private:
    std::shared_ptr<torch::optim::Optimizer> optimizer;
    std::shared_ptr<std::thread> train_thread;
    void train(const TrainParam& param);
public:
    size_t cur_epoch = 0;
    size_t cur_data_index = 0;
    std::vector<float> error;
public:
    UNet3d model;
    ~train_unet(void)
    {
        stop();
    }
    void start(const TrainParam& param);
    void stop(void);

};

#endif // TRAIN_HPP

