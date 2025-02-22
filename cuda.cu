#include "TIPL/reg/linear.hpp"
#include "TIPL/prog.hpp"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void cuda_test(){
    ;
}


namespace tipl::reg
{

template
float optimize_mi_cuda<false,tipl::out,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);
template
float optimize_mi_cuda<true,tipl::out,unsigned char,3>(
            std::shared_ptr<tipl::reg::linear_reg_param<3,unsigned char,tipl::out> > reg,
                        tipl::reg::cost_type cost_type,
                        bool& terminated);

}

bool has_cuda = true;
int gpu_count = 0;
bool distribute_gpu(void)
{
    static int cur_gpu = 0;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if(gpu_count <= 1)
        return true;
    if(cudaSetDevice(cur_gpu) != cudaSuccess)
    {
        tipl::out() << "cudaSetDevice error:" << cudaSetDevice(cur_gpu) << std::endl;
        return false;
    }
    ++cur_gpu;
    if(cur_gpu >= gpu_count)
        cur_gpu = 0;
    return true;
}

std::vector<std::string> gpu_names;
void check_cuda(std::string& error_msg)
{
    tipl::progress p("Initiating CUDA");
    int Ver;
    if(cudaGetDeviceCount(&gpu_count) != cudaSuccess ||
       cudaDriverGetVersion(&Ver) != cudaSuccess)
    {
        error_msg = "cannot obtain GPU driver and device information (CUDA ERROR:";
        error_msg += std::to_string(int(cudaGetDeviceCount(&gpu_count)));
        error_msg += "). Please update the Nvidia driver and install CUDA Toolkit.";
        return;
    }
    tipl::out() << "CUDA Driver Version: " << Ver << " CUDA Run Time Version: " << CUDART_VERSION << std::endl;
    cuda_test<<<1,1>>>();
    if(cudaPeekAtLastError() != cudaSuccess)
    {
        error_msg = "Failed to lauch cuda kernel:";
        error_msg += cudaGetErrorName(cudaGetLastError());
        error_msg += ". Please update Nvidia driver.";
        return;
    }

    tipl::out() << "Device Count:" << gpu_count << std::endl;
    for (int i = 0; i < gpu_count; i++)
    {
        tipl::out() << "Device Number:" << std::to_string(i) << std::endl;
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess)
        {
            error_msg = "Cannot obtain device information. Please update Nvidia driver";
            return;
        }
        auto arch = prop.major*10+prop.minor;
        tipl::out() << "Arch: " << arch << std::endl;
        tipl::out() << "Device name: " << prop.name << std::endl;
        tipl::out() << "Memory Size (GB): " << float(prop.totalGlobalMem >> 20)/1024.0f << std::endl;
        tipl::out() << "Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
        tipl::out() << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        tipl::out() << "Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        gpu_names.push_back(prop.name);
    }
    has_cuda = true;
}

