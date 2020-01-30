#include "CudaHost.hcu"
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

CudaHost::CudaHost()
{
    printf("CudaHost::CudaHost()\n");
    InstantiateCudaDeviceClass<<<1,1>>>(cudadevice_d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

CudaHost::~CudaHost()
{
    DeleteCudaDeviceClass << <1, 1 >> > (cudadevice_d);
    cudadevice_d = nullptr;
}

void CudaHost::Run()
{
    printf("CudaHost::Run()\n");
    RunDeviceKernels <<<1, 1 >>> (*cudadevice_d);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void InstantiateCudaDeviceClass(CudaDevice* d_ptr)
{
    printf("__global__ CudaHost::InstantiateCudaDeviceClass()\n");
    d_ptr = new CudaDevice();
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void RunDeviceKernels(CudaDevice& d_ptr)
{
    printf("__global__ CudaHost::RunDeviceKernels()\n");
    d_ptr.Run();
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void DeleteCudaDeviceClass(CudaDevice* d_ptr)
{
    printf("__global__ CudaHost::DeleteCudaDeviceClass()\n");
    delete d_ptr;
    gpuErrchk(cudaPeekAtLastError());
}
