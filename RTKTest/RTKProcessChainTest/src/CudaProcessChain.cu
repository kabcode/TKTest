
#include "CudaProcessChain.hcu"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "Reduce_kernel.hcu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__constant__ unsigned int d_vectorSize;


inline __device__ int double_kernel(int in)
{
    return in * 2;        
}

/*
inline __device__ void reduce_kernel(int* in, int* out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_vectorSize-1)
    {
        out[i] = in[i] + in[i+1];
        printf("%i + %i = %i\n", in[i], in[i+1], out[i]);
    }
}
*/

__global__ void process(int* input, int* out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d_vectorSize)
    {
        input[i] = double_kernel(input[i]);
        printf("%i\n", input[i]);
    }

    reduce_kernel(input, out, d_vectorSize);
   
}

void CUDA_process(std::vector<int> input, std::vector<int>& output)
{

    unsigned int vectorSize = input.size();
    gpuErrchk(cudaMemcpyToSymbol(d_vectorSize, &vectorSize, sizeof(unsigned int)));
    int* d_input = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_input, vectorSize * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_input, input.data(), vectorSize * sizeof(int), cudaMemcpyHostToDevice));

    int* d_out = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_out, (vectorSize-1) * sizeof(int)));

    process <<<1, 128 >>> (d_input, d_out);
    gpuErrchk(cudaDeviceSynchronize());

    output.resize(vectorSize - 1);
    gpuErrchk(cudaMemcpy(output.data(), d_out, (vectorSize - 1) * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "CUDA_process: " << output[0] << std::endl;

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_out));
}
