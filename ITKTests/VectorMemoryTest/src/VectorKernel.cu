#include <VectorKernel.hcu>

#include <cuda.h>
#include "rtkCudaUtilities.hcu"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_functions.hpp>

__constant__ int3 c_size;

__global__ void writeKernel(float* vec, int len)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i >= c_size.x || j >= c_size.y || k >= c_size.z)
		return;
	
	for(auto w = 0; w < len; ++w)
	{
		long int id = w + len * (i + c_size.x * (j + k * c_size.y));
		vec[id] = id;
	}
}


void
CUDA_writeVector(float* vec, int* size, int length)
{
	cudaMemcpyToSymbol(c_size, size, sizeof(int3));
	CUDA_CHECK_ERROR;

	dim3 dimBlock = dim3(4, 4, 4);
	int blocksInX = iDivUp(size[0], dimBlock.x);
	int blocksInY = iDivUp(size[1], dimBlock.y);
	int blocksInZ = iDivUp(size[2], dimBlock.z);
	dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);

	writeKernel <<<dimGrid, dimBlock >>> (vec, length);

	CUDA_CHECK_ERROR;
}
