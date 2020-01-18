extern inline __device__ void reduce_kernel(int* in, int* out, int size)  
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size-1)
    {
        out[i] = in[i] + in[i+1];
    }
}