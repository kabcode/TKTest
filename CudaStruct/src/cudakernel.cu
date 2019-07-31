#include "cudastruct.hcu"
#include "cudakernel.hcu"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "iostream"

template<unsigned Dimension>
__global__ void cuda_kernel(CudaStruct<Dimension> cs, float* dev_data)
{
    printf("Size: ");
    for (auto i = 0; i < Dimension; ++i)
    {
        printf("%u ", cs.size[i]);
    }

    printf("Data: ");
    for (auto i = 0; i < Dimension; ++i)
    {
        printf("%.2f ", dev_data[i*10]);
    }
    
};


template<unsigned Dimension>
void
CUDA_testkernel(CudaStruct<Dimension> cs)
{
    float* device_data;
    const size_t a_size = sizeof(float) * size_t(100*100);
    cudaMalloc((void **)&device_data, a_size);
    cudaMemcpy(device_data, cs.data, a_size, cudaMemcpyHostToDevice);

    cuda_kernel <<<1, 1 >>> (cs, device_data);
};

template void CUDA_testkernel(CudaStruct<2>);
template void CUDA_testkernel(CudaStruct<3>);