#include "cudakernel.hcu"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "iostream"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__constant__ unsigned Size[4];
__constant__ float Spacing[4];

template<unsigned int TImageDimension>
__global__ void kernel(float* indata, float* outdata)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Size[0] || j >= Size[1])
        return;

    outdata[j + i * Size[1]] = indata[j + i * Size[1]] * (Size[0] + Spacing[1]);
    printf("[%u,%u] -> %.2f -> %.2f\n", i, j, indata[j + i * Size[1]], outdata[j + i * Size[1]]);
}

template<unsigned Dimension>
void
CUDA_copyToConstant(unsigned* size, float* spacing)
{
    cudaMemcpyToSymbol(Size, size, 4 * sizeof(unsigned int));
    cudaMemcpyToSymbol(Spacing, spacing, 4 * sizeof(float));

    unsigned int checkSize[4];
    float checkSpacing[4];

    cudaMemcpyFromSymbol(checkSize, Size, 4 * sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(checkSpacing, Spacing, 4 * sizeof(float), 0, cudaMemcpyDeviceToHost);
}

template<unsigned TImageDimension>
void
CUDA_testkernel(float* indata, float* outdata)
{
   
    float* d_data;
    float* d_indata;

    unsigned int checkSize[4];
    cudaMemcpyFromSymbol(checkSize, Size, 4 * sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);

    long int outputmemory = 1;
    for (auto i = 0; i < TImageDimension; ++i)
    {
        outputmemory *= checkSize[i];
    }
    gpuErrchk(cudaMalloc((void**)&d_data, outputmemory * sizeof(float)));
    gpuErrchk(cudaMemset(d_data, 0, outputmemory * sizeof(float)));

    gpuErrchk(cudaMalloc((void**)&d_indata, outputmemory * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_indata, indata, outputmemory * sizeof(float), cudaMemcpyHostToDevice));

    auto dimBlock = dim3(16, 16);
    auto blocksInX = std::ceil(checkSize[0] / dimBlock.x);
    auto blocksInY = std::ceil(checkSize[1] / dimBlock.y);
    blocksInX = blocksInX < 1 ? 1 : blocksInX;
    blocksInY = blocksInY < 1 ? 1 : blocksInY;
    auto dimGrid = dim3(blocksInX, blocksInY);

    kernel<TImageDimension> << < dimGrid, dimBlock >> > (d_indata, d_data);
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(outdata, d_data, outputmemory * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
   
    cudaFree(d_data);
    cudaFree(d_indata);
};


template void CUDA_testkernel<2>(float*, float*);
template void CUDA_testkernel<3>(float*, float*);

template void CUDA_copyToConstant<2>(unsigned*, float*);
template void CUDA_copyToConstant<3>(unsigned*, float*);