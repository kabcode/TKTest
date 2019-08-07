#include "cudaclass.hcu"
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

__constant__ unsigned int Size[4];
__constant__ float Spacing[4];

template<unsigned int TImageDimension>
__global__ void kernel(float* indata, float* outdata)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Size[0] || j >= Size[1] )
        return;
  
    outdata[j + i * Size[1]] = indata[j + i * Size[1]] * (Size[0] + Spacing[1]);
    printf("[%u,%u] -> %.2f -> %.2f\n", i, j, indata[j + i * Size[1]], outdata[j + i * Size[1]]);
}

template<unsigned int TImageDimension>
CudaClass<TImageDimension>::CudaClass()
{
    for (auto i = 0; i < 4; ++i)
    {
        m_Size[i] = 0;
        m_Spacing[i] = 1.0f;
    }

    d_data = nullptr;
    d_indata = nullptr;
    h_data = nullptr;
}

template<unsigned int TImageDimension>
CudaClass<TImageDimension>::~CudaClass()
{
    cudaFree(d_data);
    cudaFree(d_indata);
}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::CopyConstData()
{
    //printf("Size: %u, %u, %u, %u\n", m_Size[0], m_Size[1], m_Size[2], m_Size[3]);
    //printf("Spacing: %.2f, %.2f, %.2f, %.2f\n", m_Spacing[0], m_Spacing[1], m_Spacing[2], m_Spacing[3]);

    cudaMemcpyToSymbol(Size, m_Size, 4 * sizeof(unsigned int));
    cudaMemcpyToSymbol(Spacing, m_Spacing, 4 * sizeof(float));

    unsigned int checkSize[4];
    float checkSpacing[4];

    cudaMemcpyFromSymbol(checkSize, Size, 4 * sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(checkSpacing, Spacing, 4 * sizeof(float), 0, cudaMemcpyDeviceToHost);

    //printf("Size: %u, %u, %u, %u\n", checkSize[0], checkSize[1], checkSize[2], checkSize[3]);
    //printf("Spacing: %.2f, %.2f, %.2f, %.2f\n", checkSpacing[0], checkSpacing[1], checkSpacing[2], checkSpacing[3]);
}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::SetSpacing(float * spacing)
{
    for (auto i = 0; i < TImageDimension; ++i)
    {
        m_Spacing[i] = spacing[i];
    }
}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::Run(float * indata)
{
    for (unsigned i = 0; i < m_Size[0]; ++i)
    {
        for (unsigned j = 0; j < m_Size[1]; ++j)
        {
            std::cout << indata[j + i * m_Size[1]] << " ";
        }
        std::cout << std::endl;
    }

    CopyConstData();
    long int outputmemory = 1;
    for(auto i = 0; i < TImageDimension; ++i)
    {
        outputmemory *= m_Size[i];
    }
    gpuErrchk(cudaMalloc((void**)&d_data, outputmemory * sizeof(float)));
    gpuErrchk(cudaMemset(d_data, 0, outputmemory * sizeof(float)));

    gpuErrchk(cudaMalloc((void**)&d_indata, outputmemory * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_indata, indata, outputmemory * sizeof(float), cudaMemcpyHostToDevice));

    auto dimBlock = dim3(16,16);
    auto blocksInX = std::ceil(m_Size[0] / dimBlock.x);
    auto blocksInY = std::ceil(m_Size[1] / dimBlock.y);
    blocksInX = blocksInX < 1 ? 1 : blocksInX;
    blocksInY = blocksInY < 1 ? 1 : blocksInY;
    auto dimGrid = dim3(blocksInX, blocksInY);

    kernel<TImageDimension><<< dimGrid, dimBlock >>>(d_indata, d_data);
    gpuErrchk(cudaDeviceSynchronize());
}

template<unsigned int TImageDimension>
float * CudaClass<TImageDimension>::GetData()
{
    long int outputmemory = 1;
    for (auto i = 0; i < TImageDimension; ++i)
    {
        outputmemory *= m_Size[i];
    }
    h_data = (float*) malloc(outputmemory * sizeof(float));
    cudaMemcpy(h_data, d_data, outputmemory * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return h_data;
}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::SetSize(unsigned int* size)
{
    for (auto i = 0; i < TImageDimension; ++i)
    {
        m_Size[i] = size[i];
    }
}

template class CudaClass<2>;
template class CudaClass<3>; 


