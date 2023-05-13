#include "TIPL/tipl.hpp"
#include "optiontablewidget.hpp"

template<typename image_type,typename vector_type>
void cropping_at(image_type& image,image_type& label,const vector_type& pos,int radius,float cropping_value)
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
void diffuse_light(image_type& image,tipl::uniform_dist<float>& one,float magnitude)
{
    tipl::vector<3> f(one()-0.5f,one()-0.5f,one()-0.5f);
    image_type scale(image.shape());
    auto center = tipl::vector<3>(image.shape())*0.5f;
    f.normalize();
    f *= magnitude/float(tipl::max_value(image.shape().begin(),image.shape().end()));
    for(tipl::pixel_index<3> index(image.shape());index < image.size();++index)
    {
        image[index.index()] *= std::max<float>(0.0f,1.0f + (tipl::vector<3>(index)-center)*f);
    }
}


template<typename image_type,typename vector_type>
void specular_light(image_type& image,tipl::uniform_dist<float>& one,const vector_type& center,float frequency,float mag)
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

void visual_perception_augmentation(const OptionTableWidget& options,
                          tipl::image<3>& image,
                          tipl::image<3>& label,
                          bool is_label,
                          const tipl::vector<3>& image_vs,
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


    {
        bool downsample_x = apply("downsample_x");
        bool downsample_y = apply("downsample_y");
        bool downsample_z = apply("downsample_z");
        if(downsample_x || downsample_y || downsample_z)
        {
            tipl::image<3> low_reso_image(tipl::shape<3>(float(image.shape()[0])*(downsample_x ? options.get<float>("downsample_x_ratio"): 1.0f),
                                                        float(image.shape()[1])*(downsample_y ? options.get<float>("downsample_y_ratio"): 1.0f),
                                                        float(image.shape()[2])*(downsample_z ? options.get<float>("downsample_z_ratio"): 1.0f)));
            tipl::scale(image,low_reso_image);
            tipl::scale(low_reso_image,image);
        }
    }
    if(apply("cropping"))
    {
        auto cropping_size = range(options.get<float>("cropping_size_min"),
                                   options.get<float>("cropping_size_max"));
        cropping_at(image,label,
                            random_location(image.shape(),cropping_size,1.0f - cropping_size),
                            cropping_size*float(image.width()), //radius in voxel spacing
                            range(0.0f,2.0f));
    }
    if(apply("noise"))
    {
        tipl::uniform_dist<float> noise(0.0f,options.get<float>("noise_mag"),random_seed);
        for(size_t i = 0;i < image.size();++i)
            image[i] += noise();
    }
    // lighting
    {
        if(apply("ambient"))
            ambient_light(image,one()*options.get<float>("ambient_mag"));
        if(apply("diffuse"))
            diffuse_light(image,one,options.get<float>("diffuse_mag"));
        if(apply("specular"))
            specular_light(image,one,random_location(image.shape(),0.4f,0.6f),options.get<float>("specular_freq"),options.get<float>("specular_mag"));

    }



    // rigid motion + view port
    tipl::image<3> label_out(image.shape()),image_out(image.shape());
    {
        auto resolution = range(1.0f/options.get<float>("scaling_up"),1.0f/options.get<float>("scaling_down"));
        tipl::affine_transform<float> transform = {
                    one()*float(options.get<float>("translocation_ratio"))*image.shape()[0]*image_vs[0],
                    one()*float(options.get<float>("translocation_ratio"))*image.shape()[1]*image_vs[1],
                    one()*float(options.get<float>("translocation_ratio"))*image.shape()[2]*image_vs[2],
                    one()*options.get<float>("rotation_x"),
                    one()*options.get<float>("rotation_y"),
                    one()*options.get<float>("rotation_z"),
                    resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                    resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                    resolution*range(1.0f/options.get<float>("aspect_ratio"),options.get<float>("aspect_ratio")),
                    0.0f,0.0f,0.0f};
        auto trans = tipl::transformation_matrix<float>(transform,image.shape(),image_vs,image.shape(),image_vs);


        tipl::vector<3> perspective((one()-0.5f)*options.get<float>("perspective")/float(image.shape()[0]),
                                    (one()-0.5f)*options.get<float>("perspective")/float(image.shape()[1]),
                                    (one()-0.5f)*options.get<float>("perspective")/float(image.shape()[2]));
        auto center = tipl::vector<3>(image.shape())/2.0f;


        tipl::image<3,tipl::vector<3> > displaced(image.shape());
        if(options.get<float>("lens_distortion") != 0.0f)
            lens_distortion(displaced,one()*options.get<float>("lens_distortion"));

        bool has_perspective = options.get<float>("perspective") > 0.0f;
        bool has_lens_distortion = options.get<float>("lens_distortion") > 0.0f;

        tipl::par_for(tipl::begin_index(image.shape()),tipl::end_index(image.shape()),
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
            if(is_label)
                tipl::estimate<tipl::nearest>(label,pos,label_out[index.index()]);
            else
                tipl::estimate(label,pos,label_out[index.index()]);
            tipl::estimate(image,pos,image_out[index.index()]);
        });

    }

    tipl::lower_threshold(image_out,0.0f);
    tipl::normalize(image_out);
    // background
    if(!label_out.empty() && is_label)
    {
        auto blend_fun = [&](float& src,float blend)
        {
            src += blend*std::max<float>(0.1f,1.0f-src);
        };
        if(apply("rubber_stamping"))
        {
            for(size_t pos = 0;pos < label.size();++pos)
                if(label[pos])
                    image[pos] = 0;
            tipl::image<3> background(image.shape());
            float pi2 = std::acos(-1)*2.0f;
            for(size_t iter = 0;iter < 5;++iter)
            {
                tipl::affine_transform<float> arg= {one()*image.shape()[0]*image_vs[0]*0.5f,
                                                    one()*image.shape()[1]*image_vs[1]*0.5f,
                                                    one()*image.shape()[2]*image_vs[2]*0.5f,
                                                    one()*pi2,one()*pi2,one()*pi2,
                                                    range(0.8f,1.25f),range(0.8f,1.25f),range(0.8f,1.25f),
                                                    0.0f,0.0f,0.0f};
                tipl::resample_mt(image,background,tipl::transformation_matrix<float>(arg,image.shape(),image_vs,image.shape(),image_vs));
                tipl::lower_threshold(background,0.0f);
                tipl::normalize(background,options.get<float>("rubber_stamping_mag"));
                for(size_t i = 0;i < image_out.size();++i)
                    if(!label_out[i])
                        blend_fun(image_out[i],background[i]);
            }
        }

        if(apply("perlin_texture"))
        {
            std::vector<int> p(512);
            for(size_t i = 0;i < p.size();i++)
                p[i] = i & 255;
            std::shuffle(p.begin(), p.end(),std::mt19937(random_seed));
            auto s = image.shape();
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

            tipl::normalize(background,options.get<float>("perlin_texture_mag"));
            for(size_t i = 0;i < image_out.size();++i)
                if(!label_out[i])
                    blend_fun(image_out[i],background[i]);
        }

        tipl::lower_threshold(image_out,0.0f);
        tipl::normalize(image_out);

    }



    image_out.swap(image);
    label_out.swap(label);

    }
    catch(std::runtime_error& error)
    {
        tipl::out() << "ERROR: " << error.what() << std::endl;
    }
}
