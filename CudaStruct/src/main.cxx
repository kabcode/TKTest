#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "cudakernel.hcu"

int main(int argc, char* argv[])
{
  
    unsigned int size[] = { 5,10 };
    float spacing[] = { 0.5,1 };
    CUDA_copyToConstant<2>(size, spacing);

    float data[5][10];
    for (auto i = 0; i < 5; ++i)
    {
        for (auto j = 0; j < 10; ++j)
        {
            data[i][j] = i + j;
        }
    }

    for (unsigned i = 0; i < size[0]; ++i)
    {
        for (unsigned j = 0; j < size[1]; ++j)
        {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }

    auto outdata = (float*) malloc(size[0]*size[1]*sizeof(float));

    CUDA_testkernel<2>(*data, outdata);

  
    for (unsigned i = 0; i < size[0]; ++i)
    {
        for (unsigned j = 0; j < size[1]; ++j)
        {
            std::cout << outdata[j + i * size[1]] << " ";
        }
        std::cout << std::endl;
    }

    

    unsigned int size2[] = { 4,8 };
    float spacing2[] = { 1,0.01 };
    CUDA_copyToConstant<2>(size2, spacing2);

    
    float data2[4][8];
    for (unsigned i = 0; i < size2[0]; ++i)
    {
        for (unsigned j = 0; j < size2[1]; ++j)
        {
            data2[i][j] = i * j;
        }
    }

    for (unsigned i = 0; i < size2[0]; ++i)
    {
        for (unsigned j = 0; j < size2[1]; ++j)
        {
            std::cout << data2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    auto outdata2 = (float*)malloc(size2[0] * size2[1] * sizeof(float));

    CUDA_testkernel<2>(*data2, outdata2);

    for (unsigned i = 0; i < size2[0]; ++i)
    {
        for (unsigned j = 0; j < size2[1]; ++j)
        {
            std::cout << outdata2[j + i * size2[1]] << " ";
        }
        std::cout << std::endl;
    }

    free(outdata);
    free(outdata2);
    
    return EXIT_SUCCESS;
}
