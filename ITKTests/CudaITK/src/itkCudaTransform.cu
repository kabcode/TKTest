#ifndef itkCUDATRANSFORM_CU
#define itkCUDATRANSFORM_CU
#include "itkCudaTransform.hcu"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__constant__  itk::CudaTransform<float,3,3>::DMatrix<> cMatrix;

__global__ void kernel();

namespace itk
{
    template<typename TParametersValueType, unsigned int NInputDimension, unsigned int NOutputDimension>
    void CudaTransform<TParametersValueType, NInputDimension, NOutputDimension>::FillMatrix(typename Superclass::MatrixType mat)
    {
        for (auto i = 0; i < NInputDimension; ++i)
        {
            for (auto j = 0; j < NInputDimension; ++j)
            {
                m[i][j] = mat[i][j];
            }
        }
    }

    template<typename TParametersValueType, unsigned int NInputDimension, unsigned int NOutputDimension>
    void CudaTransform<TParametersValueType, NInputDimension, NOutputDimension>::Run()
    {
        CopyMatrixOffsetToGPU();
        kernel<<<1,1>>>();
    }

    template<typename TParametersValueType, unsigned int NInputDimension, unsigned int NOutputDimension>
    void
    CudaTransform<TParametersValueType, NInputDimension, NOutputDimension>::
    CopyMatrixOffsetToGPU()
    {
        cudaMemcpyToSymbol(cMatrix, *m, sizeof(DMatrix<>));
    }

    __global__ void kernel()
    {
        printf("cMatrix: %.2f, %.2f", cMatrix.m[0][0], cMatrix.m[1][0]);
    }

}

#endif

