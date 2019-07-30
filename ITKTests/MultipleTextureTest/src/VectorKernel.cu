#include <VectorKernel.hcu>

#include <cuda.h>
#include "rtkCudaUtilities.hcu"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector_functions.hpp>
#include <cuda_runtime.h>

__constant__ int3 c_size;

template<unsigned veclen>
__global__ void useMultipleTextures(cudaTextureObject_t* tex, float* pout, int* pts)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

	float sample[veclen];
	for (unsigned int c = 0; c < veclen; c++)
		sample[c] = tex3D<float>(tex[c], i + 0.5, j + 0.5, k + 0.5);

	auto magn = 0.f;
	for (unsigned int c = 0; c < veclen; c++)
		magn += powf(sample[c], 2);

	pout[i + c_size.x * (j + k * c_size.y)] = sqrtf(magn);
}

__global__ void useSingleTexture(cudaTextureObject_t tex, float* pout)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

	float4 sample = tex3D<float4>(tex, i + 0.5, j + 0.5, k + 0.5);

	pout[i + c_size.x * (j + k * c_size.y)] = sqrtf(powf(sample.x,2)+ powf(sample.y, 2)+ powf(sample.z, 2));
}

__global__ void useNoTexture(float* pin, float* pout, int len)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

	auto a = pin[0 + len * (i + c_size.x * (j + k * c_size.y))];
	auto b = pin[1 + len * (i + c_size.x * (j + k * c_size.y))];
	auto c = pin[2 + len * (i + c_size.x * (j + k * c_size.y))];

	pout[i + c_size.x * (j + k * c_size.y)] = sqrtf(powf(a, 2) + powf(b, 2) + powf(c, 2));
	
}


void
CUDA_useMultipleTextures(float* dev_in, float* pout, int* size, int length, int* pts)
{
	
	cudaMemcpyToSymbol(c_size, size, sizeof(int3));

	int bytes = 3 * 10000 * sizeof(int);
	int* dev_pts;
	cudaMalloc(&dev_pts, bytes);
	cudaMemcpy(dev_pts, pts, bytes, cudaMemcpyHostToDevice);

	cudaTextureObject_t* tex_vol = new cudaTextureObject_t[length];
	cudaArray** volCompArrays = new cudaArray*[length];

	prepareTextureObject(size, dev_in, volCompArrays, length, tex_vol, false);
	cudaTextureObject_t* dev_tex_vol;
	cudaMalloc(&dev_tex_vol, length * sizeof(cudaTextureObject_t));
	cudaMemcpy(dev_tex_vol, tex_vol, length * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

	dim3 dimBlock = dim3(256, 1, 1);

	int blocksInX = iDivUp(size[0], dimBlock.x);
	int blocksInY = iDivUp(size[1], dimBlock.y);
	int blocksInZ = iDivUp(size[2], dimBlock.z);

	dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
	useMultipleTextures<3> <<<dimGrid, dimBlock >>> (dev_tex_vol, pout, dev_pts);

	CUDA_CHECK_ERROR;
}

void
CUDA_useSingleTexture(float* dev_in, float* pout, int* size, int length)
{

	cudaMemcpyToSymbol(c_size, size, sizeof(int3));
	
	// insert a new 0 at every fourth position
	auto ar = malloc(size[0] * size[1] * size[2] * sizeof(float4));
	for (auto k = 0; k < size[2]; ++k)
	{
		for (auto j = 0; j < size[1]; ++j)
		{
			for (auto i = 0; i < size[0]; ++i)
			{
				
			}
		}
	}

	// Allocate CUDA array in device memory
	auto channelDesc = cudaCreateChannelDesc<float4>();
	auto volExtent = make_cudaExtent(size[0], size[1], size[2]);
	cudaArray* volArray = nullptr;
	cudaMalloc3DArray((cudaArray**)& volArray, &channelDesc, volExtent);
	cudaMemcpy3DParms CopyParams = { 0 };
	CopyParams.srcPtr = make_cudaPitchedPtr((void*)dev_in, size[0] * sizeof(float4), size[0], size[1]);
	CopyParams.dstArray = volArray;
	CopyParams.extent = volExtent;
	CopyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&CopyParams);
	CUDA_CHECK_ERROR;

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = volArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
	

	dim3 dimBlock = dim3(8, 8, 8);

	int blocksInX = iDivUp(size[0], dimBlock.x);
	int blocksInY = iDivUp(size[1], dimBlock.y);
	int blocksInZ = iDivUp(size[2], dimBlock.z);

	dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
	useSingleTexture<<<dimGrid, dimBlock >>> (texObj, pout);

	CUDA_CHECK_ERROR;
}

void
CUDA_useNoTexture(float * pin, float* pout, int* size, int length)
{
	cudaMemcpyToSymbol(c_size, size, sizeof(int3));
	
	dim3 dimBlock = dim3(8, 8, 8);

	int blocksInX = iDivUp(size[0], dimBlock.x);
	int blocksInY = iDivUp(size[1], dimBlock.y);
	int blocksInZ = iDivUp(size[2], dimBlock.z);

	dim3 dimGrid = dim3(blocksInX, blocksInY, blocksInZ);
	useNoTexture <<<dimGrid, dimBlock >>> (pin, pout, length);

	CUDA_CHECK_ERROR;
}

__host__  void prepareTextureObject(int size[3],
                                    float *dev_ptr,
                                    cudaArray **&componentArrays,
                                    unsigned int nComponents,
                                    cudaTextureObject_t *tex,
                                    bool isProjections)
{
	// Create CUBLAS context
	cublasHandle_t  handle;
	cublasCreate(&handle);

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	for (unsigned int component = 0; component < nComponents; component++)
	{
		if (isProjections)
			texDesc.addressMode[component] = cudaAddressModeBorder;
		else
			texDesc.addressMode[component] = cudaAddressModeClamp;
	}
	texDesc.filterMode = cudaFilterModeLinear;

	static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaExtent volExtent = make_cudaExtent(size[0], size[1], size[2]);

	// Allocate an intermediate memory space to extract the components of the input volume
	float *singleComponent;
	int numel = size[0] * size[1] * size[2];
	cudaMalloc(&singleComponent, numel * sizeof(float));
	float one = 1.0;

	// Copy image data to arrays. The tricky part is the make_cudaPitchedPtr.
	// The best way to understand it is to read
	// http://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
	for (unsigned int component = 0; component < nComponents; component++)
	{
		// Reset the intermediate memory
		cudaMemset((void *)singleComponent, 0, numel * sizeof(float));

		// Fill it with the current component
		float * pComponent = dev_ptr + component;
		cublasSaxpy(handle, numel, &one, pComponent, nComponents, singleComponent, 1);

		// Allocate the cudaArray. Projections use layered arrays, volumes use default 3D arrays
		if (isProjections)
			cudaMalloc3DArray((cudaArray**)& componentArrays[component], &channelDesc, volExtent, cudaArrayLayered);
		else
			cudaMalloc3DArray((cudaArray**)& componentArrays[component], &channelDesc, volExtent);

		// Fill it with the current singleComponent
		cudaMemcpy3DParms CopyParams = { 0 };
		CopyParams.srcPtr = make_cudaPitchedPtr(singleComponent, size[0] * sizeof(float), size[0], size[1]);
		CopyParams.dstArray = (cudaArray*)componentArrays[component];
		CopyParams.extent = volExtent;
		CopyParams.kind = cudaMemcpyDeviceToDevice;
		cudaMemcpy3D(&CopyParams);
		CUDA_CHECK_ERROR;

		// Fill in the texture object with all this information
		resDesc.res.array.array = componentArrays[component];
		cudaCreateTextureObject(&tex[component], &resDesc, &texDesc, NULL);
		CUDA_CHECK_ERROR;
	}

	// Intermediate memory is no longer needed
	cudaFree(singleComponent);

	// Destroy CUBLAS context
	cublasDestroy(handle);
}

