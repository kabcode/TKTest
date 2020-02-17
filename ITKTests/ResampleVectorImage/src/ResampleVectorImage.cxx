// ITK includes
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkGradientImageFilter.h"
#include "HighPrecisionTimer.h"

using PixelType = float;
const unsigned int Dim3D = 3;
const unsigned int Dim2D = 2;
using VectorPixel = itk::CovariantVector<PixelType, Dim3D>;
using InputImageType = itk::Image<PixelType, Dim3D>;
using VectorImageType = itk::Image<VectorPixel, Dim3D>;
using OutputImageType = itk::Image<PixelType, Dim3D>;

using ImageReaderType = itk::ImageFileReader<InputImageType>;
using ImageWriterType = itk::ImageFileWriter<VectorImageType>;
using GradientImageFilterType = itk::GradientImageFilter<InputImageType>;
using EulerTransformType = itk::Euler3DTransform<double>;
using EulerTransformPointer = EulerTransformType::Pointer;
using ResampleImageFilterType = itk::ResampleImageFilter<InputImageType, OutputImageType>;
using ResampleVectorImageFilterType = itk::ResampleImageFilter<VectorImageType, VectorImageType>;


int main(int argc, char* argv[])
{

	auto ImageReader = ImageReaderType::New();
	ImageReader->SetFileName(argv[1]);

    auto GradientImageFilter = GradientImageFilterType::New();
    GradientImageFilter->SetInput(ImageReader->GetOutput());

	try
	{
        GradientImageFilter->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}

    auto Transform = EulerTransformType::New();
    Transform->SetIdentity();

	auto ResampleFilter = ResampleVectorImageFilterType::New();
	ResampleFilter->SetInput(GradientImageFilter->GetOutput());
	ResampleFilter->SetTransform(Transform);
	ResampleFilter->SetDefaultPixelValue(itk::NumericTraits<VectorPixel>::ZeroValue());

	ResampleFilter->SetSize(ImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
	ResampleFilter->SetOutputSpacing(ImageReader->GetOutput()->GetSpacing());

	auto OutputOrigin = Transform->GetInverseTransform()->TransformPoint(ImageReader->GetOutput()->GetOrigin());
	ResampleFilter->SetOutputOrigin(OutputOrigin);
	auto OutputDirection = Transform->GetMatrix().GetInverse() * ImageReader->GetOutput()->GetDirection().GetVnlMatrix();
    ResampleFilter->SetOutputDirection(OutputImageType::DirectionType{ OutputDirection });
    //ResampleFilter->Update();
    
    //GradientImageFilter->SetInput(ResampleFilter->GetOutput());

    std::vector<long long> durations(100);
    for (auto& duration : durations)
    {
        {
            auto Timer{ HighPrecisionTimer<TimeUnits::Milliseconds, false>(&duration) };
            ResampleFilter->Update();
        }
        ResampleFilter->Modified();
    }
    durations.erase(durations.begin(),durations.begin()+2 );
  
    auto sum = std::accumulate(durations.begin(), durations.end(), 0);
    std::cout << "MEAN: " << sum / durations.size() << std::endl;

	auto ImageWriter = ImageWriterType::New();
	ImageWriter->SetInput(ResampleFilter->GetOutput());
	ImageWriter->SetFileName("TEST.nrrd");

	try
	{
		ImageWriter->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}
