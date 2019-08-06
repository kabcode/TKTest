#include "cudaclass.hcu"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "iostream"

__constant__ int Size[4];
__constant__ int Spacing[4];

template<unsigned int TImageDimension>
CudaClass<TImageDimension>::CudaClass()
{
    for (auto i = 0; i < TImageDimension; ++i)
    {
        m_Size[i] = 0;
        m_Spacing[i] = 1.0f;
    }
}

template<unsigned int TImageDimension>
CudaClass<TImageDimension>::~CudaClass()
{
    
}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::CopyConstData()
{

}

template<unsigned int TImageDimension>
void CudaClass<TImageDimension>::SetSize(unsigned int* size )
{
    for (auto i = 0; i < TImageDimension; ++i)
    {
        m_Size[i] = size[i];
    }
}

template class CudaClass<2>;
template class CudaClass<3>; 


