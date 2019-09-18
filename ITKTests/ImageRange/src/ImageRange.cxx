// STL includes
#include <filesystem>

// ITK includes
#include "itkImageFileReader.h"
#include "itkEuler3DTransform.h"
#include "itkGradientImageFilter.h"

using PixelType = float;
const unsigned int Dim3D = 3;
using ImageType = itk::Image<PixelType, Dim3D>;

using ImageReaderType = itk::ImageFileReader<ImageType>;




int main(int argc, char* argv[])
{
	
	auto ImageReader = ImageReaderType::New();
	ImageReader->SetFileName(argv[1]);
	ImageReader->Update();

    auto GradientImageFilter = itk::GradientImageFilter<ImageType>::New();
    GradientImageFilter->SetInput(ImageReader->GetOutput());

    auto GradientImage = GradientImageFilter->GetOutput();
    GradientImage->Update();

	
	return EXIT_SUCCESS;

}
