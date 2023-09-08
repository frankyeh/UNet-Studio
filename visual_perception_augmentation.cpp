#include <unordered_map>
#include "TIPL/tipl.hpp"

template<typename image_type,typename label_type,typename vector_type>
void cropping_at(image_type& image,label_type& label,const vector_type& pos,int radius,float cropping_value)
{
    if(label.empty())
        return;
    tipl::for_each_neighbors(tipl::pixel_index<3>(pos.begin(),image.shape()),image.shape(),radius,
                                 [&](const auto& index)
    {
        if(label[index.index()])
        {
            image[index.index()] = cropping_value;
            label[index.index()] = 0;
        }
    });
}


template<typename image_type>
void ambient_light(image_type& image,float magnitude)
{
    for(size_t i = 0;i < image.size();++i)
        image[i] += magnitude;
}

template<typename image_type>
void diffuse_light(image_type& image,tipl::vector<3> f,float magnitude)
{
    auto center = tipl::vector<3>(image.shape())*0.5f;
    f.normalize();
    f *= magnitude/float(tipl::max_value(image.shape().begin(),image.shape().end()));
    for(tipl::pixel_index<3> index(image.shape());index < image.size();++index)
    {
        image[index.index()] *= std::max<float>(0.0f,1.0f + (tipl::vector<3>(index)-center)*f);
    }
}


template<typename image_type,typename vector_type>
void specular_light(image_type& image,const vector_type& center,float frequency,float mag)
{
    float b = 1.0f-mag-mag;
    frequency *= std::acos(-1)*0.5f/tipl::max_value(image.shape().begin(),image.shape().end());
    for(tipl::pixel_index<3> index(image.shape());index < image.size();++index)
    {
        image[index.index()] *= ((std::cos((tipl::vector<3>(index)-center).length()*frequency)+1.0f)*mag+b);
    }
}

template<typename image_type>
void lens_distortion(image_type& displaced,float magnitude)
{
    float radius = tipl::max_value(displaced.shape())/2;
    float radius2 = radius*radius;
    tipl::vector<3,int> center(displaced.shape());
    center /= 2;
    magnitude /= radius2;
    tipl::par_for(tipl::begin_index(displaced.shape()),
                  tipl::end_index(displaced.shape()),[&](const tipl::pixel_index<3>& pos)
    {
        tipl::vector<3> dir(pos);
        dir -= center;
        dir *= -magnitude*(dir.length2());
        displaced[pos.index()] = dir;
    });
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
void accumulate_transforms(image_type& displaced,bool has_lens_distortion,bool has_perspective,
                           const tipl::vector<3>& perspective,
                           const tipl::transformation_matrix<float>& trans)
{
    auto center = tipl::vector<3>(displaced.shape())/2.0f;
    tipl::par_for(tipl::begin_index(displaced),tipl::end_index(displaced),
        [&](const tipl::pixel_index<3>& index)
    {
        // pos now in the "retina" space
        tipl::vector<3> pos(index);
        if(has_lens_distortion)
            pos += displaced[index.index()];
        if(has_perspective)
            pos /= (perspective*(pos-center)+1.0f);
        // rigid motion + zoom + aspect ratio
        trans(pos);
        displaced[index.index()] = pos;
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

float perlin_texture(float x, float y, float z,const std::vector<int>& p)
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

void visual_perception_augmentation(std::unordered_map<std::string,float>& options,
                          tipl::image<3>& input,
                          tipl::image<3>& label,
                          bool is_label,
                          const tipl::shape<3>& image_shape,
                          const tipl::vector<3>& image_vs,
                          size_t random_seed)
{
    try{

    tipl::uniform_dist<float> one(-1.0f,1.0f,random_seed);
    auto range = [&one](float from,float to){return one()*(to-from)*0.5f+(to+from)*0.5f;};
    auto apply = [&one,&options](const char* name)
    {
        int index = int(options[name]);
        if(index == 0)
            return false;
        if(index >= 4)
            return true;
        return std::abs(one()) < float(index)*0.25f;
    };

    auto random_location = [&range](const tipl::shape<3>& sp,float from,float to)
                    {return tipl::vector<3,int>((sp[0]-1)*range(from,to),(sp[1]-1)*range(from,to),(sp[2]-1)*range(from,to));};

    tipl::image<3> output(input.shape());
    std::vector<tipl::image<3,float,tipl::pointer_container> > input_images(input.depth()/image_shape[2]),output_images(input.depth()/image_shape[2]);
    for(size_t c = 0;c < input_images.size();++c)
    {
        input_images[c] = input.alias(c*image_shape.size(),image_shape);
        output_images[c] = output.alias(c*image_shape.size(),image_shape);
    }



    {
        bool downsample_x = apply("downsample_x");
        bool downsample_y = apply("downsample_y");
        bool downsample_z = apply("downsample_z");
        if(downsample_x || downsample_y || downsample_z)
        {
            tipl::image<3> low_reso_image(tipl::shape<3>(float(image_shape[0])*(downsample_x ? options["downsample_x_ratio"]: 1.0f),
                                                        float(image_shape[1])*(downsample_y ? options["downsample_y_ratio"]: 1.0f),
                                                        float(image_shape[2])*(downsample_z ? options["downsample_z_ratio"]: 1.0f)));
            for(auto& image : input_images)
            {
                tipl::scale(image,low_reso_image);
                tipl::scale(low_reso_image,image);
            }
        }
    }

    if(apply("cropping"))
    {
        auto cropping_size = range(options["cropping_size_min"],
                                   options["cropping_size_max"])*float(image_shape.width());
        auto cropping_value = range(0.0f,2.0f);
        auto location = random_location(image_shape,cropping_size,1.0f - cropping_size);
        for(auto& image : input_images)
            cropping_at(image,label,location,cropping_size,cropping_value);
    }
    if(apply("truncation_z"))
    {
        int num_top_slices = int(std::fabs(one()*0.5f*float(label.depth())));
        int num_bottom_slices = int(std::fabs(one()*0.5f*float(label.depth())));
        auto truncate_buttom = [](auto& image,size_t num_slices)
        {
            std::fill(image.begin(),image.begin()+num_slices*image.plane_size(),0);
        };
        auto truncate_top = [](auto& image,size_t num_slices)
        {
            std::fill(image.end()-num_slices*image.plane_size(),image.end(),0);
        };
        truncate_top(label,num_top_slices);
        truncate_buttom(label,num_bottom_slices);
        for(auto& image : input_images)
        {
            truncate_top(image,num_top_slices);
            truncate_buttom(image,num_bottom_slices);
        }
    }

    if(apply("noise"))
    {
        tipl::uniform_dist<float> noise(0.0f,options["noise_mag"],random_seed);
        for(auto& image : input_images)
        for(size_t i = 0;i < image.size();++i)
            image[i] += noise();
    }
    // lighting
    if(apply("ambient"))
    {
        float ambient_magnitude = one()*options["ambient_mag"];
        for(auto& image : input_images)
            ambient_light(image,ambient_magnitude);
    }
    if(apply("diffuse"))
    {
        auto diffuse_dir = tipl::vector<3>(one()-0.5f,one()-0.5f,one()-0.5f);
        for(auto& image : input_images)
            diffuse_light(image,diffuse_dir,options["diffuse_mag"]);
    }
    if(apply("specular"))
    {
        auto location = random_location(image_shape,0.4f,0.6f);
        for(auto& image : input_images)
            specular_light(image,location,options["specular_freq"],options["specular_mag"]);
    }



    // rigid motion + view port
    tipl::image<3> output_label(image_shape);
    {
        auto resolution = range(1.0f/options["scaling_up"],1.0f/options["scaling_down"]);
        tipl::affine_transform<float> transform = {
                    one()*float(options["translocation_ratio"])*image_shape[0]*image_vs[0],
                    one()*float(options["translocation_ratio"])*image_shape[1]*image_vs[1],
                    one()*float(options["translocation_ratio"])*image_shape[2]*image_vs[2],
                    one()*options["rotation_x"],
                    one()*options["rotation_y"],
                    one()*options["rotation_z"],
                    resolution*range(1.0f/options["aspect_ratio"],options["aspect_ratio"]),
                    resolution*range(1.0f/options["aspect_ratio"],options["aspect_ratio"]),
                    resolution*range(1.0f/options["aspect_ratio"],options["aspect_ratio"]),
                    0.0f,0.0f,0.0f};
        auto trans = tipl::transformation_matrix<float>(transform,image_shape,image_vs,image_shape,image_vs);
        tipl::vector<3> perspective((one()-0.5f)*options["perspective"]/float(image_shape[0]),
                                    (one()-0.5f)*options["perspective"]/float(image_shape[1]),
                                    (one()-0.5f)*options["perspective"]/float(image_shape[2]));
        auto center = tipl::vector<3>(image_shape)/2.0f;


        tipl::image<3,tipl::vector<3> > displaced(image_shape);
        if(options["lens_distortion"] != 0.0f)
            lens_distortion(displaced,one()*options["lens_distortion"]);
        if(apply("distortion"))
        {
            size_t num = size_t(range(1.0f,options["distortion_count"]+1.0f));
            for(size_t i = 0;i < num;++i)
                create_distortion_at(displaced,random_location(image_shape,0.3f,0.7f),
                                             float(image_shape[0])*range(
                                                options["distortion_radius_min"],
                                                options["distortion_radius_max"]), // radius
                                                range(
                                                options["distortion_mag_min"],
                                                options["distortion_mag_max"]));  //magnitude
        }


        accumulate_transforms(displaced,options["lens_distortion"] > 0.0f,options["perspective"] > 0.0f,perspective,trans);


        tipl::par_for(displaced.size(),[&](size_t index)
        {
            auto pos = displaced[index];
            tipl::interpolator::linear<3> interp;
            if(!interp.get_location(image_shape,pos))
                return;
            if(is_label)
                tipl::estimate<tipl::nearest>(label,pos,output_label[index]);
            else
                interp.estimate(label,output_label[index]);
            for(int c = 0;c < input_images.size();++c)
                interp.estimate(input_images[c],output_images[c][index]);
        });

    }

    for(auto& image : output_images)
    {
        tipl::lower_threshold(image,0.0f);
        tipl::normalize(image);
    }
    // background
    if(!output_label.empty() && is_label)
    {
        if(apply("zero_background"))
        {
            for(auto& image : output_images)
                tipl::preserve(image,output_label);
            goto end;
        }
        auto blend_fun = [&](float& src,float blend)
        {
            src += blend*std::max<float>(0.1f,1.0f-src);
        };
        if(apply("rubber_stamping"))
        {
            std::vector<tipl::affine_transform<float> > args;
            float pi2 = std::acos(-1)*2.0f;
            for(size_t iter = 0;iter < 5;++iter)
                args.push_back(tipl::affine_transform<float>{one()*image_shape[0]*image_vs[0]*0.5f,
                                                    one()*image_shape[1]*image_vs[1]*0.5f,
                                                    one()*image_shape[2]*image_vs[2]*0.5f,
                                                    one()*pi2,one()*pi2,one()*pi2,
                                                    range(0.8f,1.25f),range(0.8f,1.25f),range(0.8f,1.25f),
                                                    0.0f,0.0f,0.0f});
            for(int c = 0;c < input_images.size();++c)
            {
                auto& image = input_images[c];
                auto& image_out = output_images[c];
                tipl::masking(image,label);
                tipl::image<3> background(image_shape);
                for(size_t iter = 0;iter < 5;++iter)
                {
                    tipl::resample_mt(image,background,tipl::transformation_matrix<float>(args[iter],image_shape,image_vs,image_shape,image_vs));
                    tipl::lower_threshold(background,0.0f);
                    tipl::normalize(background,options["rubber_stamping_mag"]);
                    for(size_t i = 0;i < image_out.size();++i)
                        if(!output_label[i])
                            blend_fun(image_out[i],background[i]);
                }
            }
        }

        if(apply("perlin_texture"))
        {
            std::vector<int> p(512);
            for(size_t i = 0;i < p.size();i++)
                p[i] = i & 255;
            std::shuffle(p.begin(), p.end(),std::mt19937(random_seed));
            tipl::image<3> background(image_shape);
            float zoom = range(0.005f,0.05f);
            for (int octave = 0; octave < 4; octave++)
            {
                float pow_octave = pow(0.5f, octave);
                float scale = zoom * pow_octave;
                tipl::par_for(tipl::begin_index(image_shape),
                              tipl::end_index(image_shape),[&](const tipl::pixel_index<3>& index)
                {
                    tipl::vector<3> pos(index);
                    pos *= scale;
                    float n = perlin_texture(pos[0],pos[1],pos[2],p)*pow_octave;
                    background[index.index()] += n;
                });
            }
            tipl::par_for(background.size(),[&](size_t pos)
            {
                auto v = background[pos];
                v *= 2.0f;
                background[pos] = v-std::floor(v);
            });

            tipl::normalize(background,options["perlin_texture_mag"]);
            for(auto& image : output_images)
            for(size_t i = 0;i < image.size();++i)
                if(!output_label[i])
                    blend_fun(image[i],background[i]);
        }

        for(auto& image : output_images)
        {
            tipl::lower_threshold(image,0.0f);
            tipl::normalize(image);
        }

    }

    end:

    input.swap(output);
    output_label.swap(label);

    }
    catch(std::runtime_error& error)
    {
        tipl::out() << "ERROR: " << error.what() << std::endl;
    }
}
