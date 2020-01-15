#ifndef CUDAPROCESSCHAIN_HXX
#define CUDAPROCESSCHAIN_HXX


#include "CudaProcessChain.h"
#include "CudaProcessChain.hcu"
#include <iostream>
#include <vector>

#include "itkImage.h"

template<unsigned int TSize>
auto CudaChainProcess<TSize>::Run() -> void
{
    std::cout << "Run..." << std::endl;

    std::vector<int> in{ 1,2,3,4 };
    int out = 0;
    CUDA_process(in, out);

    std::cout << "Run: " << out << std::endl;
}

template<unsigned int TSize>
auto CudaChainProcess<TSize>::setIn(int i) -> void
{
    in = i;
}


#endif // CUDAPROCESSCHAIN_HXX
