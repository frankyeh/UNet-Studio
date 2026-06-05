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
    int batch_size = 32;
    int epoch = 10000;
    float learning_rate = 0.001f;
    size_t seed = 0;
    bool is_label = true;
    bool cost_ce = true,cost_dice = true,cost_mse = true;

    std::unordered_map<std::string,float> options;
    torch::Device device = torch::kCPU;
    torch::Device test_device = torch::kCPU;
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
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          const tipl::shape<3>& model_dim,
                          const tipl::vector<3>& model_vs,
                          tipl::image<3>& input,tipl::image<3>& label);
void simulate_modality(tipl::image<3>& t1w, // store t1w or [t1w t2w]
                       const tipl::image<3>& label,
                       unsigned int max_label,
                       unsigned int seed);
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
public:
    std::string error_msg,reading_status,augmentation_status,training_status,validation_status;
private:
    std::vector<tipl::image<3> > train_image,train_label;
    std::vector<torch::Tensor> test_in_tensor,test_out_tensor;
    std::vector<char> train_image_is_template;
    size_t max_template_label = 1;
    bool test_data_ready = false,has_subject_data = false;
    std::shared_ptr<std::thread> read_images;
private:
    std::vector<tipl::image<3> > in_file,out_file;
    std::vector<size_t> in_file_read_id,in_file_seed;
    std::vector<char> file_ready;   
private:
    size_t thread_count = 1;
    std::vector<tipl::image<3> > in_data,out_data;
    std::vector<size_t> in_data_read_id;
    std::vector<char> data_ready;
    std::shared_ptr<std::thread> augmentation_thread;
    void read_file(void);
private:
    std::shared_ptr<std::thread> train_thread,validation_thread;
    void train(void);
    void validate(void);
public:
    size_t cur_epoch = 0,cur_validation_epoch = 0;
    std::mutex error_mutex;
    std::string get_status(void);

public:
    std::mutex output_model_mutex;
    UNet3d model,output_model;
    std::vector<UNet3d> other_models;
    std::string model_path;
    bool save_model_during_training = true;
    ~train_unet(void)
    {
        stop();
    }
    void join(void);
    void start(void);
    void stop(void);
};

#endif // TRAIN_HPP
