#include "zlib.h"
#include "unet.hpp"
#include "TIPL/tipl.hpp"


#include <QApplication>
#include "mainwindow.h"
#include "console.h"

tipl::program_option<tipl::out> po;
extern console_stream console;
void check_cuda(std::string& error_msg);

int tra(void);
int eval(void);
void init_application(void)
{
    QCoreApplication::setOrganizationName("LabSolver");
    QCoreApplication::setApplicationName(QString("UNet Studio"));
    if constexpr(tipl::use_cuda)
    {
        std::string cuda_msg;
        check_cuda(cuda_msg);
        if(cuda_msg.empty())
            tipl::out() << "CPU/GPU computation enabled "<< std::endl;
        else
            tipl::error() << cuda_msg;
    }
}
int run_cmd(void)
{
    if(!po.check("action"))
        return 1;
    if(!po.has("network"))
    {
        tipl::error() << "please specify --network";
        return 1;
    }
    init_application();
    if(po.get("action") == std::string("train"))
        return tra();
    if(po.get("action") == std::string("evaluate"))
        return eval();
    return 1;
}

std::string unet_studio_citation = std::string("UNet Studio version (") + __DATE__ + ", http://unet-studio.labsolver.org)";

int main(int argc, char *argv[])
{
    std::cout << unet_studio_citation << std::endl;
    if(!po.parse(argc,argv))
    {
        tipl::out() << po.error_msg << std::endl;
        return 1;
    }
    if(argc > 2)
    {
        return run_cmd();
    }

    tipl::show_prog = true;
    console.attach();
    tipl::progress prog(unet_studio_citation);
    init_application();
    QApplication a(argc, argv);
    MainWindow w;
    w.setWindowTitle(unet_studio_citation.c_str());
    w.show();
    return a.exec();

}


bool load_from_file(UNet3d& model,const char* file_name)
{
    tipl::io::gz_mat_read mat;
    tipl::out() << "load " << file_name;
    if(!mat.load_from_file(file_name))
        return false;
    std::string feature_string;
    std::vector<int> param({1,1});
    if(!mat.read("param",param) || !mat.read("feature_string",feature_string))
        return false;
    model = UNet3d(param[0],param[1],feature_string);
    mat.read("report",model->report);
    mat.read("voxel_size",model->voxel_size);
    mat.read("dimension",model->dim);
    {
        unsigned int r,c;
        if(mat.get_col_row("errors",r,c))
        {
            model->errors.resize(r*c);
            mat.read("errors",model->errors);
        }
    }
    model->train();
    int id = 0;
    for(auto& tensor : model->parameters())
    {
        unsigned int row,col;
        const auto* data = mat.read_as_type<float>((std::string("tensor")+std::to_string(id)).c_str(),row,col);
        if(!data || row*col != tensor.numel())
            return false;
        std::copy(data,data+tensor.numel(),tensor.data_ptr<float>());
        ++id;
    }
    id = 0;
    for(const auto& buffer : model->buffers())
    {
        unsigned int row,col;
        const auto* data = mat.read_as_type<float>((std::string("buffer")+std::to_string(id)).c_str(),row,col);
        if(!data || row*col != buffer.numel())
            continue;
        if(buffer.scalar_type() == torch::kFloat)
            std::copy(data,data+buffer.numel(),buffer.data_ptr<float>());
        if(buffer.scalar_type() == torch::kLong)
            std::copy(data,data+buffer.numel(),buffer.data_ptr<int64_t>());
        ++id;
    }
    /*
    tipl::shape<3> image_volume(6,6,6);
    std::vector<float> data(image_volume.size());
    for(size_t i = 0;i < data.size();++i)
        data[i] = i;

    model->train(false);
    model->print_layers();

    auto cn = model->encoding[0]->begin();
    auto re = cn+1;
    auto bn = re+1;
    auto cn2 = bn+1;
    auto max_pool3d = model->encoding[1]->begin();
    auto upsampling = model->up[0]->begin();
    auto data_tensor = torch::from_blob(&data[0],{1,1,int(image_volume[2]),int(image_volume[1]),int(image_volume[0])});
    //auto out_tensor = upsampling->forward(max_pool3d->forward(bn->forward(re->forward(cn->forward(data_tensor)))));
    auto out_tensor = cn2->forward(bn->forward(re->forward(cn->forward(data_tensor))));
    tipl::out() << out_tensor.sizes();


    tipl::ml3d::network n;
    //n << new conv_3d(1,8) << (new relu(8)) << new batch_norm_3d(8) << new max_pool_3d(8) << new upsample_3d(8);
    n << new tipl::ml3d::conv_3d(1,8)
      << new tipl::ml3d::relu(8)
      << new tipl::ml3d::batch_norm_3d(8)
      << new tipl::ml3d::conv_3d(8,8);
    n.print(std::cout);

    auto dim = image_volume;
    n.init_image(dim);
    auto params = n.parameters();


    auto ptr1 = model->parameters()[0].data_ptr<float>();
    auto ptr2 = model->parameters()[1].data_ptr<float>();
    auto ptr3 = model->parameters()[2].data_ptr<float>();
    auto ptr4 = model->parameters()[3].data_ptr<float>();
    auto ptr5 = model->parameters()[4].data_ptr<float>();
    auto ptr6 = model->parameters()[5].data_ptr<float>();

    std::copy(ptr1,ptr1+8*3*3*3,params[0].first);
    std::copy(ptr2,ptr2+8,params[1].first);
    std::copy(ptr3,ptr3+8,params[2].first);
    std::copy(ptr4,ptr4+8,params[3].first);
    std::copy(ptr5,ptr5+8*8*3*3*3,params[4].first);
    std::copy(ptr6,ptr6+8,params[5].first);

    auto ptr = n.forward(&data[0]);

    for(size_t i = 0;i < out_tensor.numel();++i)
        tipl::out() << out_tensor.data_ptr<float>()[i] << "\t" << ptr[i] << "\t" << ptr[i] - out_tensor.data_ptr<float>()[i];
    */
    return true;
}
bool save_to_file(UNet3d& model,const char* file_name)
{
    tipl::io::gz_mat_write mat(file_name);
    if(!mat)
        return false;
    mat.write("feature_string",model->feature_string);
    mat.write("report",model->report);
    mat.write("voxel_size",model->voxel_size);
    mat.write("dimension",model->dim);
    mat.write("errors",model->errors,3);
    mat.write("param",std::vector<int>({model->in_count,model->out_count}));
    int id = 0;
    for(const auto& tensor : model->parameters())
    {
        auto cpu_tensor = tensor.to(torch::kCPU);
        mat.write((std::string("tensor")+std::to_string(id)).c_str(),cpu_tensor.data_ptr<float>(),cpu_tensor.numel()/cpu_tensor.sizes().front(),cpu_tensor.sizes().front());
        ++id;
    }
    id = 0;
    for(const auto& buffer : model->buffers())
    {
        auto cpu_buffer = buffer.to(torch::kCPU);
        if(!cpu_buffer.numel())
            continue;
        if (cpu_buffer.scalar_type() == torch::kFloat)
            mat.write((std::string("buffer") + std::to_string(id)).c_str(),
                          cpu_buffer.data_ptr<float>(),
                          cpu_buffer.numel() / cpu_buffer.sizes().front(),
                          cpu_buffer.sizes().front());
        else
            if (cpu_buffer.scalar_type() == torch::kLong)
                mat.write((std::string("buffer") + std::to_string(id)).c_str(),cpu_buffer.data_ptr<int64_t>(),1,1);
            else
                tipl::warning() << "buffer not saved due to type " << int(cpu_buffer.scalar_type());
        ++id;
    }
    return true;
}
std::string show_structure(const UNet3d& model)
{
    std::ostringstream out;
    std::vector<int> ks;
    auto features = model->parse_feature_string(ks);
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
