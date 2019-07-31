#include <cstdlib>
#include <algorithm>

#include "cudastruct.hcu"
#include "cudakernel.hcu"

int main(int argc, char* argv[])
{
    CudaStruct<2> s;
    s.size[0] = 100;
    s.size[1] = 100;

    float arr[100];
    std::fill(arr, arr + 100, 12);
    s.data = arr;

    CUDA_testkernel(s);


    return EXIT_SUCCESS;
}
