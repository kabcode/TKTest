#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "cudaclass.hcu"

auto main(int argc, char* argv[]) -> int
{
    auto cudaclass = new CudaClass<2>;
    auto cudaclass2 = new CudaClass<2>;

    unsigned int size[] = { 5,10 };
    cudaclass->SetSize(size);

    float spacing[] = { 0.5,1 };
    cudaclass->SetSpacing(spacing);

    unsigned int size2[] = { 4,8 };
    cudaclass2->SetSize(size2);

    float spacing2[] = { 1,1 };
    cudaclass2->SetSpacing(spacing2);

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

    cudaclass->Run(*data);

    const auto outdata = cudaclass->GetData();

    
    for (unsigned i = 0; i < size[0]; ++i)
    {
        for (unsigned j = 0; j < size[1]; ++j)
        {
            std::cout << outdata[j + i * size[1]] << " ";
        }
        std::cout << std::endl;
    }


   

  

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

    cudaclass2->Run(*data2);

    const auto outdata2 = cudaclass2->GetData();


    for (unsigned i = 0; i < size2[0]; ++i)
    {
        for (unsigned j = 0; j < size2[1]; ++j)
        {
            std::cout << outdata2[j + i * size2[1]] << " ";
        }
        std::cout << std::endl;
    }


    delete cudaclass;
    delete cudaclass2;

    return EXIT_SUCCESS;
}
