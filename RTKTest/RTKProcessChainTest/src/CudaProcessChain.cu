
#include "CudaProcessChain.hcu"
#include <iostream>

__managed__ int d_result;

inline __device__ void double_kernel(int* in, int& out)
{
    printf("double_kernel: %i, %i\n", in[1], in[2]);
    out = 5;
}
inline __device__ int sum_kernel(int* in)
{
    printf("sum_kernel: %i, %i\n", in[1], in[1000]);
    return 2;
}


__global__ void process(int* input)
{
    printf("process: %i, %i\n", input[1], input[2]);
    int inter;
    double_kernel(input, inter);
    printf("inter: %i\n", inter);
    d_result = sum_kernel(&inter);
    printf("d_result: %i\n", d_result);
}

void CUDA_process(std::vector<int> input, int& output)
{
    int* d_input;
    cudaMalloc((void**)&d_input, input.size() * sizeof(int));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice);

    process<<<1,1>>>(d_input);
    cudaDeviceSynchronize();

    output = d_result;

    std::cout << "CUDA_process: " << output << std::endl;

   
}
