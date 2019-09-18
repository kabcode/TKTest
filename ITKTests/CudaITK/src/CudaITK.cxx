#include "itkCudaImage.h"


#include "itkCudaTransform.hcu"

int main(int argc, char* argv[])
{
    auto mat = itk::Matrix<float, 2,2>();
    mat[0][0] = 1.2345345;
    mat[0][1] = 2.2345345;
    mat[1][0] = 3.2345345;
    mat[1][1] = 3.2345345;

    auto transform = itk::CudaTransform<float,2, 2>::New();
    transform->FillMatrix(mat);
    transform->Run();

    return EXIT_SUCCESS;
}