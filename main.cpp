#include <filesystem>
#include "zlib.h"
#include "unet.hpp"
#include "TIPL/tipl.hpp"


#include <QApplication>
#include <QMessageBox>
#include "mainwindow.h"
#include "console.h"

tipl::program_option<tipl::out> po;

std::string find_full_path(const std::string& name)
{
    std::filesystem::path file_path(name);

    std::filesystem::path app_dir_file = std::filesystem::path(QCoreApplication::applicationDirPath().toUtf8().constData())/file_path;
    if(std::filesystem::exists(app_dir_file))
        return app_dir_file.string();

    std::filesystem::path cwd_file = std::filesystem::current_path()/file_path;
    if(std::filesystem::exists(cwd_file))
        return cwd_file.string();

    return name;
}
std::vector<std::string> seg_template_list;
std::vector<std::vector<std::string> > atlas_file_name_list;
bool load_file_name(void)
{
    namespace fs = std::filesystem;

    fs::path dir = fs::path(QCoreApplication::applicationDirPath().toUtf8().constData())/"atlas";
    if(!fs::exists(dir) && !fs::exists(dir = fs::current_path()/"atlas"))
        return false;

    std::vector<std::string> name_list(tipl::get_directories(dir));

    auto get_rank = [](const std::string& d)
    {
        int rank = 0;
        for(const auto& k : {"human","chimpanzee","rhesus","marmoset","rat","mouse"})
        {
            if(d.find(k) != std::string::npos)
                return rank;
            ++rank;
        }
        return rank;
    };

    std::stable_sort(name_list.begin(),name_list.end(),[&](const std::string& a,const std::string& b)
    {
        return get_rank(a) < get_rank(b);
    });

    for(const auto& name : name_list)
    {
        fs::path t_dir = dir/name;
        fs::path tissue_file = t_dir/(name+"_tissue.nii.gz");

        if(!fs::exists(tissue_file))
            continue;

        seg_template_list.push_back(tissue_file.string());

        std::vector<std::string> atlas_list,file_list;
        for(const auto& entry : fs::directory_iterator(t_dir))
            if(entry.is_regular_file() && tipl::ends_with(entry.path().filename().string(),{".nii",".nii.gz"}))
                atlas_list.push_back(entry.path().filename().string());

        std::sort(atlas_list.begin(),atlas_list.end());

        for(const auto& each : atlas_list)
            if(each.substr(0,each.find('_')) != name)
                file_list.push_back((t_dir/each).string());

        atlas_file_name_list.push_back(std::move(file_list));
    }

    return !seg_template_list.empty();
}

extern console_stream console;
void check_cuda(std::string& error_msg);
int tra(void);
int eval(void);
bool init_application(void)
{
    QCoreApplication::setOrganizationName("LabSolver");
    QCoreApplication::setApplicationName(QString("UNet Studio"));
    if constexpr(tipl::use_cuda)
    {
        tipl::out() << "Checking CUDA functions"<< std::endl;
        std::string cuda_msg;
        check_cuda(cuda_msg);
        if(cuda_msg.empty())
            tipl::out() << "CPU/GPU computation enabled "<< std::endl;
        else
            tipl::error() << cuda_msg;
    }
    if(!load_file_name())
        return tipl::error() << "cannot find template and atlases",false;
    return true;
}
int run_cmd(void)
{
    if(!init_application())
        return 1;
    if(!po.check("action"))
        return 1;
    if(!po.has("network"))
    {
        tipl::error() << "please specify --network";
        return 1;
    }
    if(po.get("action") == std::string("train"))
        return tra();
    if(po.get("action") == std::string("evaluate"))
        return eval();
    return 1;
}

std::string unet_studio_citation = std::string("UNet Studio version (") + __DATE__ + ", http://unet-studio.labsolver.org)";



int main(int argc, char *argv[])
{
    tipl::out() << unet_studio_citation << std::endl;
    if(!po.parse(argc,argv))
        return tipl::out() << po.error_msg,1;
    if(argc > 2)
    {
        QCoreApplication a(argc,argv);
        return run_cmd();
    }
    tipl::show_prog = true;
    console.attach();
    tipl::progress prog(unet_studio_citation);
    QApplication a(argc, argv);
    if(!init_application())
    {
        QMessageBox::critical(nullptr,"ERROR","cannot find template");
        return 1;
    }
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
    std::vector<std::vector<int> > features_down,features_up;
    tipl::ml3d::parse_feature_string(model->feature_string,model->in_count,features_down,features_up,ks);

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
