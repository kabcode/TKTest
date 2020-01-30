
#include "rtkCudaGradientImageFilter.h"
#include <itkCudaImage.h>
#include "itkGradientImageFilter.h"
#include "rtkAdditiveGaussianNoiseImageFilter.h"

#include "rtkCudaCropImageFilter.h"
#include "rtkCudaForwardProjectionImageFilter.h"


using Pixeltype = float;
const unsigned int Dimension = 3;

using CudaImageType = itk::CudaImage<Pixeltype, Dimension>;
using GradientImageFilterType = itk::GradientImageFilter < CudaImageType, float, float>;

int main(int argc, char* argv[])
{
    auto GradientImageFilter = GradientImageFilterType::New();

    auto f = rtk::AdditiveGaussianNoiseImageFilter<CudaImageType>::New();
    f->SetMean(2.0);

    auto FixedImage = CudaImageType::New();
    auto MovingImage = CudaImageType::New();

    auto c = rtk::CudaCropImageFilter::New();
    c->SetInput(FixedImage);

    auto p = rtk::CudaForwardProjectionImageFilter<>::New();
    p->SetInput(FixedImage);

    auto g = rtk::CudaGradientImageFilter<>::New();
    g->SetInput(FixedImage);


    std::cout << g->GetNameOfClass() << std::endl;

    return EXIT_SUCCESS;
}