#include "itkMultiplyImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

using ImageType = itk::Image<float, 3>;
using VectorImageType = itk::Image<itk::CovariantVector<float, 3>, 3>;

int main(int argc, char* argv[])
{
    auto Reader = itk::ImageFileReader<ImageType>::New();
    Reader->SetFileName(argv[1]);

    auto GradientFilter = itk::GradientImageFilter<ImageType>::New();
    GradientFilter->SetInput(Reader->GetOutput());

    const float Multiplier = 1;
    

    auto MultiplyFilter = itk::MultiplyImageFilter<VectorImageType, ImageType>::New();
    MultiplyFilter->SetInput1(GradientFilter->GetOutput());
    MultiplyFilter->SetConstant2(Multiplier);

    auto Writer = itk::ImageFileWriter<VectorImageType>::New();
    Writer->SetFileName("MultiplyVectorImage2.nrrd");
    Writer->SetInput(MultiplyFilter->GetOutput());
    Writer->Update();

	
	return EXIT_SUCCESS;
}