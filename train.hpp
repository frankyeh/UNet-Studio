#ifndef TRAIN_HPP
#define TRAIN_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"

struct TrainParam{
    std::vector<std::string> image_file_name,test_image_file_name;
    std::vector<std::string> label_file_name,test_label_file_name;
    int batch_size = 1;
    int epoch = 10000;
    float learning_rate = 0.001f;
    int output_model_type = 0;
    bool is_label = true;
    torch::Device device = torch::kCPU;
};


bool save_to_file(UNet3d& model,const char* file_name);
bool load_from_file(UNet3d& model,const char* file_name);
std::string show_structure(const UNet3d& model);
bool get_label_info(const std::string& label_name,int& out_count,bool& is_label);
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          tipl::vector<3>& vs);
class OptionTableWidget;
void visual_perception_augmentation(const OptionTableWidget& options,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          bool is_label,
                          const tipl::shape<3>& image_shape,
                          const tipl::vector<3>& image_vs,
                          size_t random_seed);
class train_unet{
public:
    TrainParam param;
    OptionTableWidget* option = nullptr;
public:
    bool aborted = false;
    bool pause = false;
    bool running = false;
    std::string error_msg,status;
private:
    std::vector<tipl::image<3> > in_data,out_data;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> read_file_thread;
    std::vector<torch::Tensor> test_in_tensor,test_out_tensor;
    std::vector<tipl::image<3,char> > test_mask;
    bool test_data_ready = false;
    void read_file(void);
private:
    std::vector<torch::Tensor> in_tensor,out_tensor;
    std::vector<bool> tensor_ready;
    std::shared_ptr<std::thread> prepare_tensor_thread;
    void prepare_tensor(void);
private:
    std::shared_ptr<torch::optim::Optimizer> optimizer;
    std::shared_ptr<std::thread> train_thread;
    void train(void);
public:
    size_t cur_epoch = 0;
    std::vector<float> error;
    std::vector<std::vector<float> > test_error_foreground,test_error_background;
public:
    UNet3d model,output_model;
    ~train_unet(void)
    {
        stop();
    }
    void start(void);
    void stop(void);

};

#endif // TRAIN_HPP

