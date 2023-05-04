#include "train.hpp"
#include "optiontablewidget.hpp"
bool get_label_info(const std::string& label_name,int& out_count,bool& is_label)
{
    tipl::io::gz_nifti nii;
    if(!nii.load_from_file(label_name))
        return false;
    if(nii.dim(4) != 1)
        out_count = nii.dim(4);
    if(nii.is_integer())
    {
        is_label = true;
        if(nii.dim(4) == 1)
        {
            tipl::image<3,short> labels;
            nii >> labels;
            out_count = tipl::max_value(labels);
        }
    }
    else
    {
        tipl::image<3,float> labels;
        nii >> labels;
        is_label = tipl::is_label_image(labels);
        if(nii.dim(4) == 1)
            out_count = (is_label ? tipl::max_value(labels) : 1);
    }

    if(out_count > 128)
    {
        out_count = 1;
        is_label = false;
    }
    return true;
}
bool read_image_and_label(const std::string& image_name,
                          const std::string& label_name,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          tipl::vector<3>& vs)
{
    tipl::matrix<4,4,float> image_t((tipl::identity_matrix()));
    if(!tipl::io::gz_nifti::load_from_file(image_name.c_str(),image,vs,image_t))
        return false;

    float sd = float(tipl::standard_deviation(image));
    if(sd != 0.0f)
        image *= 1.0/sd;

    tipl::io::gz_nifti nii;
    tipl::shape<3> label_shape;
    if(!nii.load_from_file(label_name))
        return false;
    nii.get_image_dimension(label_shape);
    if(nii.dim(4) > 1)
    {
        label.resize(label_shape);
        for(size_t index = 1;index <= nii.dim(4);++index)
        {
            tipl::image<3> I;
            nii >> I;
            tipl::par_for(I.size(),[&](size_t pos)
            {
                if(I[pos])
                    label[pos] = index;
            });
        }
    }
    else
    {
        nii >> label;
    }
    tipl::matrix<4,4,float> label_t((tipl::identity_matrix()));
    nii.get_image_transformation(label_t);
    if(image.shape() != label_shape || label_t != image_t)
    {
        tipl::out() << "spatial transform label file to image file space" << std::endl;
        tipl::image<3> new_label(image.shape());
        tipl::resample_mt<tipl::nearest>(label,new_label,tipl::from_space(image_t).to(label_t));
        label.swap(new_label);
    }
    return true;
}

template<typename image_type>
void diffuse_light(image_type& image_out,tipl::uniform_dist<float>& one,float magnitude)
{
    tipl::vector<3> f(one()-0.5f,one()-0.5f,one()-0.5f);
    image_type scale(image_out.shape());
    auto center = tipl::vector<3>(image_out.shape())*0.5f;
    f.normalize();
    f *= magnitude/float(tipl::max_value(image_out.shape().begin(),image_out.shape().end()));
    for(tipl::pixel_index<3> index(image_out.shape());index < image_out.size();++index)
    {
        tipl::vector<3> pos(index);
        pos -= center;
        image_out[index.index()] *= std::max<float>(0.0f,1.0f + pos*f);
    }
}


template<typename image_type>
void specular_light(image_type& image_out,tipl::uniform_dist<float>& one,const tipl::vector<3,int>& center,float frequency,float mag)
{
    float b = 1.0f-mag-mag;
    frequency *= std::acos(-1)*0.5f/tipl::max_value(image_out.shape().begin(),image_out.shape().end());
    for(tipl::pixel_index<3> index(image_out.shape());index < image_out.size();++index)
    {
        tipl::vector<3> pos(index.begin());
        pos -= center;
        image_out[index.index()] *= ((std::cos(pos.length()*frequency)+1.0f)*mag+b);
    }
}
template<typename image_type>
void ghost(image_type& image_out,int shift,float magnitude,bool direction)
{
    tipl::image<3> ghost = image_out;
    ghost *= magnitude;
    if(direction)
    {
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(shift,0,0));
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(-shift,0,0));
    }
    else
    {
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(0,shift,0));
        tipl::draw<false>(ghost,image_out,tipl::vector<3,int>(0,-shift,0));
    }
}
template<typename image_type>
void create_distortion_at(image_type& displaced,const tipl::vector<3,int>& center,float radius,float magnitude)
{
    auto radius_5 = radius*magnitude;
    auto pi_2_radius = std::acos(-1)/radius;
    tipl::for_each_neighbors(tipl::pixel_index<3>(center.begin(),displaced.shape()),displaced.shape(),radius,[&](const auto& pos)
    {
        tipl::vector<3> dir(pos);
        dir -= center;
        auto length = dir.length();
        if(length > radius)
            return;
        dir *= -radius_5*std::sin(length*pi_2_radius)/length;
        displaced[pos.index()] += dir;
    });
}

template<typename image_type>
void create_occlusion_at(image_type& image,image_type& label,const tipl::vector<3,int>& pos,int radius,float occlusion_value)
{
    tipl::for_each_neighbors(tipl::pixel_index<3>(pos.begin(),image.shape()),image.shape(),radius,
                                 [&](const auto& index)
    {
        image[index.index()] = occlusion_value;
        if(!label.empty())
            label[index.index()] = 0;
    });
}



inline float lerp(float t, float a, float b)	{return a + t * (b - a);}
inline float fade(float t) 			{return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);}
float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

float perlin_noise(float x, float y, float z,const std::vector<int>& p)
{
    int xi = (int)floor(x) & 255;
    int yi = (int)floor(y) & 255;
    int zi = (int)floor(z) & 255;

    float xf = x - floor(x);
    float yf = y - floor(y);
    float zf = z - floor(z);

    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    int aaa = p[p[p[xi] + yi] + zi];
    int aba = p[p[p[xi] + yi + 1] + zi];
    int aab = p[p[p[xi] + yi] + zi + 1];
    int abb = p[p[p[xi] + yi + 1] + zi + 1];
    int baa = p[p[p[xi + 1] + yi] + zi];
    int bba = p[p[p[xi + 1] + yi + 1] + zi];
    int bab = p[p[p[xi + 1] + yi] + zi + 1];
    int bbb = p[p[p[xi + 1] + yi + 1] + zi + 1];

    float x1 = lerp(u, grad(aaa, xf, yf, zf),
                        grad(baa, xf - 1, yf, zf));
    float x2 = lerp(u, grad(aba, xf, yf - 1, zf),
                        grad(bba, xf - 1, yf - 1, zf));
    float y1 = lerp(v, x1, x2);

    x1 = lerp(u, grad(aab, xf, yf, zf - 1),
                  grad(bab, xf - 1, yf, zf - 1));
    x2 = lerp(u, grad(abb, xf, yf - 1, zf - 1),
                  grad(bbb, xf - 1, yf - 1, zf - 1));
    float y2 = lerp(v, x1, x2);

    return lerp(w, y1, y2);
}


void load_image_and_label(const OptionTableWidget& options,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          bool is_label,
                          const tipl::vector<3>& image_vs,
                          const tipl::vector<3>& template_vs,
                          const tipl::shape<3>& template_shape,
                          size_t random_seed)
{
    try{

    tipl::uniform_dist<float> one(-1.0f,1.0f,random_seed);
    auto range = [&one](float from,float to){return one()*(to-from)*0.5f+(to+from)*0.5f;};
    auto apply = [&one,&options](const char* name)
    {
        int index = options.get<int>(name);
        if(index == 0)
            return false;
        if(index >= 4)
            return true;
        return std::abs(one()) < float(index)*0.25f;
    };
    auto random_location = [&range](const tipl::shape<3>& sp,float from,float to)
                    {return tipl::vector<3,int>((sp[0]-1)*range(from,to),(sp[1]-1)*range(from,to),(sp[2]-1)*range(from,to));};

    auto resolution = range(options.get<float>("scaling_down"),options.get<float>("scaling_up"));
    tipl::affine_transform<float> transform = {
                one()*float(options.get<float>("translocation_ratio"))*template_shape[0]*template_vs[0],
                one()*float(options.get<float>("translocation_ratio"))*template_shape[1]*template_vs[1],
                one()*float(options.get<float>("translocation_ratio"))*template_shape[2]*template_vs[2],
                one()*options.get<float>("rotation_x"),
                one()*options.get<float>("rotation_y"),
                one()*options.get<float>("rotation_z"),
                resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                one()*options.get<float>("affine"),
                one()*options.get<float>("affine"),
                one()*options.get<float>("affine")};

    tipl::image<3,tipl::vector<3> > displaced(template_shape);


    if(apply("deformation"))
    {
        size_t num = size_t(range(1.0f,options.get<int>("foci_count")+1.0f));
        for(size_t i = 0;i < num;++i)
            create_distortion_at(displaced,random_location(image.shape(),0.3f,0.7f),
                                     float(image.shape()[0])*range(
                                        options.get<float>("foci_radius_min"),
                                        options.get<float>("foci_radius_max")), // radius
                                        range(
                                        options.get<float>("deformation_mag_min"),
                                        options.get<float>("deformation_mag_max")));  //magnitude
    }


    if(apply("occlusion"))
    {
        auto count = options.get<float>("occlusion_count");
        for(size_t i = 0;i < count;++i)
        {
            auto occlusion_size = range(options.get<float>("occlusion_size_min"),
                                      options.get<float>("occlusion_size_max"));
            create_occlusion_at(image,label,
                          random_location(image.shape(),occlusion_size,1.0f - occlusion_size),
                                occlusion_size*float(image.width()),
                                range(0.0f,1.0f));
        }
    }

    if(!label.empty())
    {
        tipl::image<3> label_out(template_shape);
        if(is_label)
            tipl::compose_displacement_with_affine<tipl::nearest>(label,label_out,tipl::transformation_matrix<float>(transform,template_shape,template_vs,image.shape(),image_vs),displaced);
        else
            tipl::compose_displacement_with_affine(label,label_out,tipl::transformation_matrix<float>(transform,template_shape,template_vs,image.shape(),image_vs),displaced);
        label_out.swap(label);
    }

    tipl::image<3> image_out(template_shape);
    tipl::compose_displacement_with_affine(image,image_out,tipl::transformation_matrix<float>(transform,template_shape,template_vs,image.shape(),image_vs),displaced);

    bool downsample_x = apply("downsample_x");
    bool downsample_y = apply("downsample_y");
    bool downsample_z = apply("downsample_z");
    if(downsample_x || downsample_y || downsample_z)
    {
        tipl::image<3> low_reso_image(tipl::shape<3>(float(template_shape[0])*(downsample_x ? options.get<float>("downsample_x_ratio"): 1.0f),
                                                    float(template_shape[1])*(downsample_y ? options.get<float>("downsample_y_ratio"): 1.0f),
                                                    float(template_shape[2])*(downsample_z ? options.get<float>("downsample_z_ratio"): 1.0f)));
        tipl::scale(image_out,low_reso_image);
        low_reso_image.swap(image_out);
    }
    if(apply("blurring"))
    {
        auto count = options.get<float>("blurring_count");
        for(size_t i = 0;i < count;++i)
            tipl::filter::gaussian(image_out);
    }

    if(apply("ambient"))
        image_out += range(0.0f,options.get<float>("ambient_mag"));
    if(apply("diffuse"))
        diffuse_light(image_out,one,options.get<float>("diffuse_mag"));
    if(apply("specular"))
        specular_light(image_out,one,random_location(image_out.shape(),0.4f,0.6f),options.get<float>("specular_freq"),options.get<float>("specular_mag"));

    tipl::normalize(image_out);
    tipl::lower_threshold(image_out,0.0f);


    if(image_out.shape() != template_shape)
    {
        tipl::image<3> original_reso_image(template_shape);
        tipl::scale(image_out,original_reso_image);
        original_reso_image.swap(image_out);
    }


    if(!label.empty())
    {
        if(apply("self_replication"))
        {
            tipl::image<3> background(template_shape);
            {
                tipl::affine_transform<float> arg= {one()*30.0f,one()*30.0f,one()*30.0f,
                                                    one()*2.0f,one()*2.0f,one()*2.0f,
                                                    0.25f,0.25f,0.25f,
                                                    0.0f,0.0f,0.0f};
                tipl::resample_mt(image,background,tipl::transformation_matrix<float>(arg,template_shape,template_vs,image.shape(),image_vs));
            }
            tipl::normalize(background,options.get<float>("self_replication_mag"));

            if(!is_label)
                image_out += background;
            else
            for(size_t i = 0;i < image_out.size();++i)
                if(!label[i])
                    image_out[i] += background[i];
        }

        if(apply("perlin_noise"))
        {
            std::vector<int> p(512);
            for(size_t i = 0;i < p.size();i++)
                p[i] = i & 255;
            std::shuffle(p.begin(), p.end(),std::mt19937(random_seed));
            auto s = template_shape;
            tipl::image<3> background(s);
            float zoom = range(0.005f,0.05f);
            for (int octave = 0; octave < 4; octave++)
            {
                float pow_octave = pow(0.5f, octave);
                float scale = zoom * pow_octave;
                tipl::par_for(tipl::begin_index(s),
                              tipl::end_index(s),[&](const tipl::pixel_index<3>& index)
                {
                    tipl::vector<3> pos(index);
                    pos *= scale;
                    float n = perlin_noise(pos[0],pos[1],pos[2],p)*pow_octave;
                    background[index.index()] += n;
                });
            }
            tipl::par_for(background.size(),[&](size_t pos)
            {
                auto v = background[pos];
                v *= 2.0f;
                background[pos] = v-std::floor(v);
            });

            tipl::normalize(background,options.get<float>("perlin_noise_mag"));
            if(!is_label)
                image_out += background;
            else
            for(size_t i = 0;i < image_out.size();++i)
                if(!label[i])
                    image_out[i] += background[i];
        }

        if(apply("background_scale"))
        {
            float scale = range(options.get<float>("background_scale_min"),options.get<float>("background_scale_max"));
            for(size_t i = 0;i < image_out.size();++i)
                if(!label[i])
                    image_out[i] *= scale;
        }
    }


    if(apply("noise"))
    {
        tipl::uniform_dist<float> noise(0.0f,options.get<float>("noise_mag"),random_seed);
        for(size_t i = 0;i < image_out.size();++i)
            image_out[i] += noise();
    }

    image_out.swap(image);

    }
    catch(std::runtime_error& error)
    {
        tipl::out() << "ERROR: " << error.what() << std::endl;
    }
}

std::vector<size_t> get_label_count(const tipl::image<3>& label,size_t out_count)
{
    std::vector<size_t> sum(out_count);
    for(size_t i = 0;i < label.size();++i)
        if(label[i])
        {
            auto v = label[i]-1;
            if(v < out_count)
                ++sum[v];
        }
    return sum;
}
tipl::shape<3> unet_inputsize(const tipl::shape<3>& s);

void train_unet::read_file(void)
{
    int thread_count = std::min<int>(8,std::thread::hardware_concurrency());
    in_data = std::vector<tipl::image<3> >(thread_count);
    out_data = std::vector<tipl::image<3> >(thread_count);
    data_ready = std::vector<bool>(thread_count,false);

    test_data_ready = false;
    test_in_tensor.clear();
    test_out_tensor.clear();
    test_mask.clear();

    read_file_thread.reset(new std::thread([=]()
    {
        std::vector<tipl::image<3> > train_image,train_label;
        std::vector<tipl::vector<3> > image_vs;
        // prepare training data
        {
            status = "reading input data";
            tipl::progress prog("read training data");
            for(int read_id = 0;read_id < param.image_file_name.size();++read_id)
            {
                tipl::out() << "reading " << param.image_file_name[read_id] << std::endl;
                tipl::out() << "reading " << param.label_file_name[read_id] << std::endl;
                tipl::image<3> new_image,new_label;
                tipl::vector<3> new_vs;

                if(!read_image_and_label(param.image_file_name[read_id],
                                         param.label_file_name[read_id],
                                         new_image,new_label,new_vs))
                {
                    error_msg = "cannot read image or label data for ";
                    error_msg += std::filesystem::path(param.image_file_name[read_id]).filename().string();
                    aborted = true;
                    return;
                }
                tipl::normalize(new_image);
                if(!param.is_label)
                    tipl::normalize(new_label);
                train_image.push_back(std::move(new_image));
                train_label.push_back(std::move(new_label));
                image_vs.push_back(new_vs);
            }
            if(train_image.empty())
            {
                error_msg = "no training image";
                aborted = true;
                return;
            }
            tipl::out() << "a total of " << train_image.size() << " training dataset" << std::endl;
        }
        //prepare test data
        {
            tipl::progress prog("read test data");
            for(int read_id = 0;read_id < param.test_image_file_name.size();++read_id)
            {
                tipl::out() << "reading " << param.test_image_file_name[read_id] << std::endl;
                tipl::out() << "reading " << param.test_label_file_name[read_id] << std::endl;
                tipl::image<3> new_image,new_label;
                tipl::vector<3> new_vs;
                if(!read_image_and_label(param.test_image_file_name[read_id],
                                         param.test_label_file_name[read_id],
                                         new_image,new_label,new_vs))
                {
                    error_msg = "cannot read image or label data for ";
                    error_msg += std::filesystem::path(param.test_image_file_name[read_id]).filename().string();
                    aborted = true;
                    return;
                }
                if(new_image.shape() != model->dim)
                {
                    tipl::image<3> I(model->dim),I2(model->dim);
                    auto shift = tipl::vector<3,int>(model->dim)-tipl::vector<3,int>(new_image.shape());
                    tipl::draw(new_image,I,shift);
                    tipl::draw(new_label,I2,shift);
                    new_image.swap(I);
                    new_label.swap(I2);
                }
                tipl::image<3,char> mask(model->dim);
                if(model->out_count > 1)
                {
                    for(size_t pos = 0;pos < mask.size();++pos)
                        if(new_label[pos])
                            mask[pos] = 1;
                    expand_label_to_dimension(new_label,model->out_count);
                }
                else
                    tipl::normalize(new_label);
                tipl::normalize(new_image);
                test_in_tensor.push_back(torch::from_blob(&new_image[0],
                    {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_out_tensor.push_back(torch::from_blob(&new_label[0],
                    {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device));
                test_mask.push_back(std::move(mask));
            }
            if(test_in_tensor.empty())
            {
                error_msg = "no test data";
                aborted = true;
                return;
            }
            tipl::out() << "a total of " << test_mask.size() << " testing dataset" << std::endl;
            test_data_ready = true;
        }

        model->voxel_size = image_vs[0];
        size_t seed_base = model->total_training_count;
        tipl::out() << "vision simulation starts" << std::endl;
        tipl::par_for(thread_count,[&](size_t thread)
        {
            size_t seed = thread + seed_base;
            while(!aborted)
            {
                while(data_ready[thread] || pause)
                {
                    using namespace std::chrono_literals;
                    status = "network training";
                    std::this_thread::sleep_for(100ms);
                    if(aborted)
                        return;
                }
                auto read_id = seed % train_image.size();
                if(!train_image[read_id].size())
                    continue;
                in_data[thread] = train_image[read_id];
                out_data[thread] = train_label[read_id];
                load_image_and_label(*option,in_data[thread],out_data[thread],param.is_label,image_vs[read_id],model->voxel_size,model->dim,seed);
                if(model->out_count > 1)
                    tipl::expand_label_to_dimension(out_data[thread],model->out_count);
                data_ready[thread] = true;
                seed += thread_count;
            }
        });
    }));
}

void train_unet::prepare_tensor(void)
{
    const int thread_count = std::min<int>(4,std::thread::hardware_concurrency());
    in_tensor = std::vector<torch::Tensor>(thread_count);
    out_tensor = std::vector<torch::Tensor>(thread_count);
    tensor_ready = std::vector<bool>(thread_count,false);
    prepare_tensor_thread.reset(new std::thread([=](){
        try{
            tipl::par_for(thread_count,[&](size_t thread)
            {
                size_t i = thread;
                while(!aborted)
                {
                    size_t data_index = i%data_ready.size();
                    while(!data_ready[data_index] || tensor_ready[thread] || pause)
                    {
                        using namespace std::chrono_literals;
                        status = "image deformation and transformation";
                        std::this_thread::sleep_for(100ms);
                        if(aborted)
                            return;
                    }
                    in_tensor[thread] = torch::from_blob(&in_data[data_index][0],
                        {1,model->in_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device);
                    out_tensor[thread] = torch::from_blob(&out_data[data_index][0],
                        {1,model->out_count,int(model->dim[2]),int(model->dim[1]),int(model->dim[0])}).to(param.device);
                    tensor_ready[thread] = true;
                    data_ready[data_index] = false;
                    i += thread_count;
                }
            });
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error in preparing tensor:") + error.what();
            pause = aborted = true;
        }
        catch(...)
        {
            pause = aborted = true;
        }
        tipl::out() << error_msg << std::endl;
    }));
}

void train_unet::train(void)
{
    error = std::vector<float>(param.epoch);
    test_error_foreground = std::vector<std::vector<float> >(param.test_image_file_name.size());
    for(auto& each : test_error_foreground)
        each.resize(param.epoch);
    test_error_background = test_error_foreground;
    cur_epoch = 0;

    output_model = UNet3d(1,model->out_count,model->feature_string);
    output_model->to(param.device);
    output_model->copy_from(*model);

    train_thread.reset(new std::thread([=](){
        struct exist_guard
        {
            bool& running;
            exist_guard(bool& running_):running(running_){}
            ~exist_guard() { running = false; }
        } guard(running);

        try{

            size_t cur_data_index = 0;
            size_t best_epoch = 0;

            optimizer.reset(new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(param.learning_rate)));

            while(!test_data_ready || pause)
            {
                if(aborted)
                    return;
                status = "tensor allocation";
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(100ms);
            }
            tipl::out() << "begin training networks" << std::endl;
            for (; cur_epoch < param.epoch && !aborted;cur_epoch++)
            {
                if(param.learning_rate != optimizer->defaults().get_lr())
                {
                    tipl::out() << "set learning rate to " << param.learning_rate << std::endl;
                    optimizer->defaults().set_lr(param.learning_rate);
                }

                std::ostringstream out;
                {
                    model->eval();
                    out << "epoch:" << cur_epoch << " testing error:";
                    for(size_t i = 0;i < test_in_tensor.size();++i)
                    {
                        tipl::image<3> dif_map(model->dim.multiply(tipl::shape<3>::z,model->out_count));
                        std::memcpy(&dif_map[0],(model->forward(test_in_tensor[i])-test_out_tensor[i]).to(torch::kCPU).data_ptr<float>(),dif_map.size()*sizeof(float));

                        auto thread_count = std::min<int>(8,std::thread::hardware_concurrency());
                        std::vector<double> sum_foreground(thread_count);
                        std::vector<double> sum_background(thread_count);
                        const auto& mask = test_mask[i];
                        tipl::par_for(thread_count,[&](size_t thread_id)
                        {
                            for(size_t out_base = 0;out_base < dif_map.size();out_base += mask.size())
                            for(size_t pos = thread_id;pos < mask.size();pos += thread_count)
                            {
                                float v = std::fabs(dif_map[out_base+pos]);
                                if(mask[pos])
                                    sum_foreground[thread_id] += v;
                                else
                                    sum_background[thread_id] += v;
                            }
                        });
                        test_error_foreground[i][cur_epoch] = tipl::sum(sum_foreground)/float(dif_map.size());
                        test_error_background[i][cur_epoch] = tipl::sum(sum_background)/float(dif_map.size());
                        out << test_error_foreground[i][cur_epoch] << " " << test_error_background[i][cur_epoch] << " ";
                    }
                    if(test_error_foreground[0][best_epoch] + test_error_background[0][best_epoch] <=
                       test_error_foreground[0][cur_epoch]  + test_error_background[0][cur_epoch])
                        best_epoch = cur_epoch;

                    if(param.output_model_type == 0 || best_epoch == cur_epoch)
                        output_model->copy_from(*model);

                    model->train();
                }

                float sum_error = 0.0f;
                for(size_t b = 0;b < param.batch_size && !aborted;++b,++cur_data_index)
                {
                    size_t data_index = cur_data_index%tensor_ready.size();
                    while(!tensor_ready[data_index] || pause)
                    {
                        if(aborted)
                            return;
                        status = "tensor allocation";
                        using namespace std::chrono_literals;
                        std::this_thread::sleep_for(100ms);
                    }
                    status = "training";
                    torch::Tensor loss = torch::l1_loss(model->forward(in_tensor[data_index]),out_tensor[data_index]);
                    sum_error += loss.item().toFloat();
                    loss.backward();
                    tensor_ready[data_index] = false;
                }

                {
                    optimizer->step();
                    optimizer->zero_grad();
                    model->total_training_count += param.batch_size;
                    out << " training error:" << (error[cur_epoch] = sum_error/float(param.batch_size)) << std::endl;

                }
                tipl::out() << out.str();
            }
        }
        catch(const c10::Error& error)
        {
            error_msg = std::string("error in training:") + error.what();
        }
        catch(...)
        {
        }
        tipl::out() << error_msg << std::endl;
        pause = aborted = true;
        model->copy_from(*output_model);
        status = "complete";
    }));
}
void train_unet::start(void)
{
    stop();
    status = "initializing";
    model->to(param.device);
    model->train();
    pause = aborted = false;
    running = true;
    error_msg.clear();
    read_file();
    prepare_tensor();
    train();
}
void train_unet::stop(void)
{
    pause = aborted = true;
    if(read_file_thread.get())
    {
        read_file_thread->join();
        read_file_thread.reset();
    }
    if(prepare_tensor_thread.get())
    {
        prepare_tensor_thread->join();
        prepare_tensor_thread.reset();
    }
    if(train_thread.get())
    {
        train_thread->join();
        train_thread.reset();
    }
}

