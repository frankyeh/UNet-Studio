#include "TIPL/tipl.hpp"
#include "TIPL/cuda/mem.hpp"
#include "TIPL/cuda/basic_image.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T1,typename T2>
__global__ void cropping_at_kernel(T1 image,T2 label,tipl::vector<3> pos,float radius,float cropping_value)
{
    TIPL_FOR(index,image.size())
    {
        tipl::vector<3> dir(tipl::pixel_index<3>(index,image.shape()));
        dir -= pos;
        if(dir[0] > radius || dir[1] > radius || dir[2] > radius)
            return;
        auto length = dir.length();
        if(length > radius)
            return;
        if(label[index])
        {
            image[index] = cropping_value;
            label[index] = 0;
        }
    }
}
template<typename image_type,typename label_type,typename vector_type>
inline void cropping_at(image_type& image,label_type& label,const vector_type& pos,float radius,float cropping_value)
{
    TIPL_RUN(cropping_at_kernel,image.size())
            (tipl::make_shared(image),tipl::make_shared(label),pos,radius,cropping_value);
}

template<typename T>
__global__ void truncate_top_kernel(T from,size_t slices)
{
    TIPL_FOR(index,slices*from.plane_size())
    {
        from[from.size()-1-index] = 0;
    }
}
template<typename T>
inline void truncate_top(T& from,size_t slices)
{
    TIPL_RUN(truncate_top_kernel,slices*from.plane_size())
            (tipl::make_shared(from),slices);
}

template<typename T>
__global__ void truncate_bottom_kernel(T from,size_t slices)
{
    TIPL_FOR(index,slices*from.plane_size())
    {
        from[index] = 0;
    }
}
template<typename T>
inline void truncate_buttom(T& from,size_t slices)
{
    TIPL_RUN(truncate_bottom_kernel,slices*from.plane_size())
            (tipl::make_shared(from),slices);
}

#include <curand_kernel.h>

template<typename T>
__global__ void add_noise_kernel(T from,float noise_level)
{
    TIPL_FOR(index,from.size())
    {
        curandState state;
        curand_init(0, index, 0, &state);
        from[index] += noise_level*curand_uniform(&state);
    }
}
template<typename T>
inline void add_noise(T& from,float noise_level)
{
    TIPL_RUN(add_noise_kernel,from.size())
            (tipl::make_shared(from),noise_level);
}

template<typename T>
__global__ void diffuse_light_kernel(T from,tipl::vector<3> f,tipl::vector<3> center,float magnitude)
{
    TIPL_FOR(index,from.size())
    {
        from[index] *= std::max<float>(0.0f,1.0f + (tipl::vector<3>(tipl::pixel_index<3>(index,from.shape()))-center)*f);
    }
}

template<typename T>
inline void diffuse_light_cuda(T& from,tipl::vector<3> f,float magnitude)
{
    auto center = tipl::vector<3>(from.shape())*0.5f;
    f.normalize();
    f *= magnitude/float(tipl::max_value(from.shape().begin(),from.shape().end()));
    TIPL_RUN(diffuse_light_kernel,from.size())
            (tipl::make_shared(from),f,center,magnitude);
}


template<typename T>
__global__ void specular_light_kernel(T from,tipl::vector<3> center,float frequency,float mag,float b)
{
    TIPL_FOR(index,from.size())
    {
        from[index] *= ((std::cos((tipl::vector<3>(tipl::pixel_index<3>(index,from.shape()))-center).length()*frequency)+1.0f)*mag+b);
    }
}

template<typename T>
inline void specular_light_cuda(T& from,tipl::vector<3> center,float frequency,float mag)
{
    float b = 1.0f-mag-mag;
    frequency *= std::acos(-1)*0.5f/tipl::max_value(from.shape().begin(),from.shape().end());
    TIPL_RUN(specular_light_kernel,from.size())
            (tipl::make_shared(from),center,frequency,mag,b);
}
template<typename T>
__global__ void lens_distortion_kernel(T displaced,tipl::vector<3> center,float magnitude)
{
    TIPL_FOR(index,displaced.size())
    {
        tipl::vector<3> dir(tipl::pixel_index<3>(index,displaced.shape()));
        dir -= center;
        dir *= -magnitude*(dir.length2());
        displaced[index] = dir;
    }
}
template<typename image_type>
inline void lens_distortion_cuda(image_type& displaced,float magnitude)
{
    float radius = tipl::max_value(displaced.shape())/2;
    float radius2 = radius*radius;
    tipl::vector<3,int> center(displaced.shape());
    center /= 2;
    magnitude /= radius2;
    TIPL_RUN(lens_distortion_kernel,displaced.size())
            (tipl::make_shared(displaced),center,magnitude);
}

template<typename T>
__global__ void create_distortion_at_kernel(T displaced,tipl::vector<3> center,float radius,float radius_5,float pi_2_radius)
{
    TIPL_FOR(index,displaced.size())
    {
        tipl::vector<3> dir(tipl::pixel_index<3>(index,displaced.shape()));
        dir -= center;
        if(dir[0] > radius || dir[1] > radius || dir[2] > radius)
            return;
        auto length = dir.length();
        if(length > radius)
            return;
        dir *= -radius_5*std::sin(length*pi_2_radius)/length;
        displaced[index] += dir;
    }
}
template<typename image_type>
inline void create_distortion_at_cuda(image_type& displaced,const tipl::vector<3,int>& center,float radius,float magnitude)
{
    auto radius_5 = radius*magnitude;
    auto pi_2_radius = std::acos(-1)/radius;
    TIPL_RUN(create_distortion_at_kernel,displaced.size())
            (tipl::make_shared(displaced),center,radius,radius_5,pi_2_radius);
}

template<typename T>
__global__ void accumulate_transforms_kernel(T displaced,bool has_lens_distortion,bool has_perspective,
                                             tipl::vector<3> center,
                                             tipl::vector<3> perspective,
                                             tipl::transformation_matrix<float> trans)
{
    TIPL_FOR(index,displaced.size())
    {
        // pos now in the "retina" space
        tipl::vector<3> pos(tipl::pixel_index<3>(index,displaced.shape()));
        if(has_lens_distortion)
            pos += displaced[index];
        if(has_perspective)
            pos /= (perspective*(pos-center)+1.0f);
        // rigid motion + zoom + aspect ratio
        trans(pos);
        displaced[index] = pos;
    }
}




template<typename image_type>
inline void accumulate_transforms_cuda(image_type& displaced,bool has_lens_distortion,bool has_perspective,
                           const tipl::vector<3>& perspective,
                           const tipl::transformation_matrix<float>& trans)
{
    auto center = tipl::vector<3>(displaced.shape())/2.0f;
    TIPL_RUN(accumulate_transforms_kernel,displaced.size())
            (tipl::make_shared(displaced),has_lens_distortion,has_perspective,center,perspective,trans);
}


template<typename T1,typename T2,typename T3>
__global__ void blend_kernel(T1 image,T2 label,T3 background)
{
    TIPL_FOR(index,image.size())
    {
        if(!label[index])
            image[index] += background[index]*std::max<float>(0.1f,1.0f-image[index]);
    }
}



__INLINE__ float lerp_cuda(float t, float a, float b)	{return a + t * (b - a);}
__INLINE__ float fade_cuda(float t) 			{return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);}
__INLINE__ float grad_cuda(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

template<typename T>
__INLINE__ float perlin_texture_cuda(float x, float y, float z,T p)
{
    int xi = (int)floor(x) & 255;
    int yi = (int)floor(y) & 255;
    int zi = (int)floor(z) & 255;

    float xf = x - floor(x);
    float yf = y - floor(y);
    float zf = z - floor(z);

    float u = fade_cuda(xf);
    float v = fade_cuda(yf);
    float w = fade_cuda(zf);

    int aaa = p[p[p[xi] + yi] + zi];
    int aba = p[p[p[xi] + yi + 1] + zi];
    int aab = p[p[p[xi] + yi] + zi + 1];
    int abb = p[p[p[xi] + yi + 1] + zi + 1];
    int baa = p[p[p[xi + 1] + yi] + zi];
    int bba = p[p[p[xi + 1] + yi + 1] + zi];
    int bab = p[p[p[xi + 1] + yi] + zi + 1];
    int bbb = p[p[p[xi + 1] + yi + 1] + zi + 1];

    float x1 = lerp_cuda(u, grad_cuda(aaa, xf, yf, zf),
                        grad_cuda(baa, xf - 1, yf, zf));
    float x2 = lerp_cuda(u, grad_cuda(aba, xf, yf - 1, zf),
                        grad_cuda(bba, xf - 1, yf - 1, zf));
    float y1 = lerp_cuda(v, x1, x2);

    x1 = lerp_cuda(u, grad_cuda(aab, xf, yf, zf - 1),
                  grad_cuda(bab, xf - 1, yf, zf - 1));
    x2 = lerp_cuda(u, grad_cuda(abb, xf, yf - 1, zf - 1),
                  grad_cuda(bbb, xf - 1, yf - 1, zf - 1));
    float y2 = lerp_cuda(v, x1, x2);

    return lerp_cuda(w, y1, y2);
}

template<typename T,typename T2>
__global__ void perlin_texture_kernel(T background,T2 p,float scale,float pow_octave)
{
    TIPL_FOR(index,background.size())
    {
        tipl::vector<3> pos(tipl::pixel_index<3>(index,background.shape()));
        pos *= scale;
        background[index] += perlin_texture_cuda(pos[0],pos[1],pos[2],p.data())*pow_octave;
    }
}

template<typename T>
__global__ void perlin_texture_kernel2(T background)
{
    TIPL_FOR(index,background.size())
    {
        float v = background[index];
        v *= 2.0f;
        background[index] = v-std::floor(v);
    }
}



void visual_perception_augmentation_cuda(std::unordered_map<std::string,float>& options,
                          tipl::image<3>& input_,
                          tipl::image<3>& label_,
                          bool is_label,
                          const tipl::shape<3>& image_shape,
                          const tipl::vector<3>& image_vs,
                          size_t random_seed)
{
    try{

    {
        int gpu_count = 1;
        if(cudaGetDeviceCount(&gpu_count) != cudaSuccess ||
            cudaSetDevice(gpu_count-1) != cudaSuccess)
            tipl::out() << "cudaSetDevice error:" << cudaSetDevice(gpu_count-1) << std::endl;
    }

    tipl::device_image<3> input = input_;
    tipl::device_image<3> label = label_;

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

    tipl::device_image<3> output(input.shape());
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
            tipl::device_image<3> low_reso_image(tipl::shape<3>(float(image_shape[0])*(downsample_x ? options["downsample_x_ratio"]: 1.0f),
                                                        float(image_shape[1])*(downsample_y ? options["downsample_y_ratio"]: 1.0f),
                                                        float(image_shape[2])*(downsample_z ? options["downsample_z_ratio"]: 1.0f)));
            for(auto& image : input_images)
            {
                tipl::scale_cuda(image,low_reso_image);
                tipl::scale_cuda(low_reso_image,image);
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
        float noise_mag = options["noise_mag"];
        for(auto& image : input_images)
            add_noise(image,noise_mag);
    }
    // lighting
    if(apply("ambient"))
    {
        float ambient_magnitude = one()*options["ambient_mag"];
        for(auto& image : input_images)
            tipl::add_constant_cuda(image,ambient_magnitude);
    }
    if(apply("diffuse"))
    {
        auto diffuse_dir = tipl::vector<3>(one()-0.5f,one()-0.5f,one()-0.5f);
        for(auto& image : input_images)
            diffuse_light_cuda(image,diffuse_dir,options["diffuse_mag"]);
    }
    if(apply("specular"))
    {
        auto location = random_location(image_shape,0.4f,0.6f);
        for(auto& image : input_images)
            specular_light_cuda(image,location,options["specular_freq"],options["specular_mag"]);
    }

    // rigid motion + view port
    tipl::device_image<3> output_label(image_shape);
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


        tipl::device_image<3,tipl::vector<3> > displaced(image_shape);
        if(options["lens_distortion"] != 0.0f)
            lens_distortion_cuda(displaced,one()*options["lens_distortion"]);

        if(apply("distortion"))
        {
            size_t num = size_t(range(1.0f,options["distortion_count"]+1.0f));
            for(size_t i = 0;i < num;++i)
                create_distortion_at_cuda(displaced,random_location(image_shape,0.3f,0.7f),
                                             float(image_shape[0])*range(
                                                options["distortion_radius_min"],
                                                options["distortion_radius_max"]), // radius
                                                range(
                                                options["distortion_mag_min"],
                                                options["distortion_mag_max"]));  //magnitude
        }


        accumulate_transforms_cuda(displaced,options["lens_distortion"] > 0.0f,options["perspective"] > 0.0f,perspective,trans);

        if(is_label)
            tipl::compose_mapping_cuda<tipl::nearest>(label,displaced,output_label);
        else
            tipl::compose_mapping_cuda(label,displaced,output_label);

        for(size_t c = 0;c < output_images.size();++c)
            tipl::compose_mapping_cuda(input_images[c],displaced,output_images[c]);
    }


    for(auto& image : output_images)
    {
        tipl::lower_threshold_cuda(image,0.0f);
        tipl::normalize_cuda(image);
    }

    // background
    if(!output_label.empty() && is_label)
    {

        if(apply("zero_background"))
        {
            for(auto& image : output_images)
                tipl::preserve_cuda(image,output_label);
            goto end;
        }


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

                tipl::masking_cuda(image,label);

                tipl::device_image<3> background(image_shape);
                for(size_t iter = 0;iter < 5;++iter)
                {
                    tipl::resample_cuda(image,background,tipl::transformation_matrix<float>(args[iter],image_shape,image_vs,image_shape,image_vs));
                    tipl::lower_threshold_cuda(background,0.0f);
                    tipl::normalize_cuda(background,options["rubber_stamping_mag"]);
                    TIPL_RUN(blend_kernel,image_out.size())
                            (tipl::make_shared(image_out),tipl::make_shared(output_label),tipl::make_shared(background));
                }
            }
        }

        if(apply("perlin_texture"))
        {
            std::vector<int> p(512);
            for(size_t i = 0;i < p.size();i++)
                p[i] = i & 255;
            std::shuffle(p.begin(), p.end(),std::mt19937(random_seed));

            tipl::device_vector<int> p_device = p;

            tipl::device_image<3> background(image_shape);
            float zoom = range(0.005f,0.05f);
            for (int octave = 0; octave < 4; octave++)
            {
                float pow_octave = pow(0.5f, octave);
                float scale = zoom * pow_octave;
                TIPL_RUN(perlin_texture_kernel,background.size())
                        (tipl::make_shared(background),tipl::make_shared(p_device),scale,pow_octave);
            }

            TIPL_RUN(perlin_texture_kernel2,background.size())
                    (tipl::make_shared(background));

            tipl::normalize_cuda(background,options["perlin_texture_mag"]);
            for(auto& image : output_images)
                TIPL_RUN(blend_kernel,image.size())
                (tipl::make_shared(image),tipl::make_shared(output_label),tipl::make_shared(background));

        }

        for(auto& image : output_images)
        {
            tipl::lower_threshold_cuda(image,0.0f);
            tipl::normalize_cuda(image);
        }

    }

    end:

    output.buf().copy_to(input_.buf());
    output_label.buf().copy_to(label_.buf());

    }
    catch(std::runtime_error& error)
    {
        tipl::out() << "ERROR: " << error.what() << std::endl;
    }
    if(cudaSetDevice(0) != cudaSuccess)
        tipl::out() << "cudaSetDevice error:" << cudaSetDevice(0) << std::endl;
}
