#include "CudaDevice.hcu"
#include "iostream"

CudaDevice::CudaDevice()
{
    printf("CudaDevice::CudaDevice()\n");
    numbers[2] =100;
}

void CudaDevice::Run()
{
    printf("CudaDevice()::Run()\n");
    unsigned int number_size = 5;
    RevealNumber<<<1,32>>>(numbers, number_size);
}

 __global__ void RevealNumber(int* number, unsigned int number_size)
{
    printf("CudaDevice()::RevealNumber()\n");
    unsigned int idx = blockDim.x * gridDim.x + threadIdx.x;
    if (idx < number_size)
    {
        printf("Here comes: %i", number[idx]);
    }
}

