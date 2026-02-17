#ifndef TRAIN_HPP
#define TRAIN_HPP
#include <string>
#include <vector>
#include "zlib.h"
#include "unet.hpp"



struct training_param{
    std::vector<std::string> image_file_name,test_image_file_name;
    std::vector<std::string> label_file_name,test_label_file_name;
    std::vector<float> subject_label_weight;
    int batch_size = 8;
    int epoch = 4000;
    float learning_rate = 0.00025f;
    bool is_label = true;
    std::unordered_map<std::string,float> options;
    torch::Device device = torch::kCPU;
    inline void set_weight(std::string w)
    {
        std::istringstream in(w);
        auto label_weight = std::vector<float>((std::istream_iterator<float>(in)),std::istream_iterator<float>());
        tipl::multiply_constant(label_weight,1.0f/(tipl::sum(label_weight)));

        subject_label_weight = label_weight;
    }
};


bool save_to_file(UNet3d& model,const char* file_name);
bool load_from_file(UNet3d& model,const char* file_name);
std::string show_structure(const UNet3d& model);
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          size_t in_count,
                          tipl::image<3>& input,
                          tipl::image<3>& label,
                          tipl::shape<3>& image_shape);
void visual_perception_augmentation(std::unordered_map<std::string,float>& options,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          bool is_label,
                          const tipl::shape<3>& image_shape,
                          size_t random_seed);
class train_unet{
public:
    training_param param;
public:
    bool aborted = false;
    bool pause = false;
    bool running = false;
    std::string error_msg,reading_status,augmentation_status,training_status;
private:
    std::vector<tipl::image<3> > train_image,train_label;
    std::vector<bool> train_image_is_template;
    std::vector<torch::Tensor> test_in_tensor,test_out_tensor;
    bool test_data_ready = false;
    std::shared_ptr<std::thread> read_images;
private:
    std::vector<tipl::image<3> > in_file,out_file;
    std::vector<size_t> in_file_read_id,in_file_seed;
    std::vector<bool> file_ready;

private:
    size_t thread_count = 1;
    std::vector<tipl::image<3> > in_data,out_data;
    std::vector<size_t> in_data_read_id;
    std::vector<bool> data_ready;
    std::shared_ptr<std::thread> augmentation_thread;
    void read_file(void);
private:
    std::shared_ptr<std::thread> train_thread;
    void train(void);
public:
    size_t cur_epoch = 0;
    std::vector<std::string> test_error_name;
    std::vector<std::vector<float> > test_error;
    std::mutex error_mutex;
    bool save_error_to(const char* file_name);
    std::string get_status(void);
private:
    UNet3d test_model;
public:
    std::mutex output_model_mutex;
    UNet3d model,output_model;
    std::vector<UNet3d> other_models;
    ~train_unet(void)
    {
        stop();
    }
    void join(void);
    void start(void);
    void stop(void);

};

#endif // TRAIN_HPP

