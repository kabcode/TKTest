
#include "itkImageFileReader.h"
#include "itkRTKForwardProjectionImageFilter.h"
#include "itkAdvGradientDifferenceImageToImageMetric.h"
#include "itkEuler3DTransform.h"
#include "itkMultiplyImageFilter.h"
#include <chrono>

using PixelType = float;
static constexpr unsigned int Dimension = 3;
using CudaImageType = itk::CudaImage<PixelType, Dimension>;
using CudaImagepointer = CudaImageType::Pointer;
using ImageType = itk::Image<PixelType, Dimension>;
using ImagePointer = ImageType::Pointer;


int main(int argc, char* argv[])
{

	auto Reader = itk::ImageFileReader<CudaImageType>::New();
	Reader->SetFileName(argv[1]);

	auto Image = Reader->GetOutput();
	Image->Update();
	/*
	auto CudaImage = CudaImageType::New();
	CudaImage->SetSpacing(Image->GetSpacing());
	CudaImage->SetDirection(Image->GetDirection());
	CudaImage->SetOrigin(Image->GetOrigin());
	CudaImage->SetRegions(Image->GetLargestPossibleRegion());
	CudaImage->SetPixelContainer(Image->GetPixelContainer());
	*/
	auto Transform = itk::Euler3DTransform<>::New();
	Transform->SetIdentity();

	auto Interpolator = itk::LinearInterpolateImageFunction<itk::Image<itk::CovariantVector<float,3>,3>>::New();
	

	auto AGD = itk::AdvGradientDifferenceImageToImageMetric<CudaImageType, CudaImageType>::New();
	AGD->SetMovingImage(Image);
	AGD->SetFixedImage(Image);
	AGD->SetFixedImageRegion(Image->GetLargestPossibleRegion());
	AGD->SetTransform(Transform);
	AGD->SetInterpolator(Interpolator);
	AGD->Initialize();

	auto params = Transform->GetParameters();
	auto start = std::chrono::high_resolution_clock::now();
	for (auto i = -10; i < 10; ++i)
	{
		params[3] = i;
		for (auto j = -10; j < 10; ++j)
		{
			params[4] = j;
			AGD->GetValue(params);
		}
	}
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
	
	std::cout << Image->GetBufferPointer() << std::endl;
	
	auto ImageFileWriter = itk::ImageFileWriter<CudaImageType>::New();
	ImageFileWriter->SetFileName("Output.nrrd");
	ImageFileWriter->SetInput(Image);

	try
	{
		auto start = std::chrono::high_resolution_clock::now();
		ImageFileWriter->Update();
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
	}
	catch (itk::ExceptionObject& EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}