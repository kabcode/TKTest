#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkTranslationTransform.h"

int main(int argc, char* argv[])
{
	using ImageType = itk::Image< double, 2 >;
	using ReaderType = itk::ImageFileReader<ImageType>;
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " imageFile1 imageFile2" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << argv[1] << std::endl;
	std::cout << argv[2] << std::endl;

	ReaderType::Pointer fixedReader = ReaderType::New();
	fixedReader->SetFileName(argv[1]);
	fixedReader->Update();
	ReaderType::Pointer movingReader = ReaderType::New();
	movingReader->SetFileName(argv[2]);
	movingReader->Update();
	ImageType::Pointer fixedImage = fixedReader->GetOutput();
	ImageType::Pointer movingImage = movingReader->GetOutput();
	using MetricType = itk::MeanSquaresImageToImageMetric < ImageType, ImageType >;
	using InterpolatorType = itk::LinearInterpolateImageFunction<ImageType, double >;
	using TransformType = itk::TranslationTransform < double, 2 >;
	MetricType::Pointer metric = MetricType::New();
	TransformType::Pointer transform = TransformType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	interpolator->SetInputImage(fixedImage);
	metric->SetFixedImage(fixedImage);
	metric->SetMovingImage(movingImage);
	metric->SetFixedImageRegion(fixedImage->GetLargestPossibleRegion());
	metric->SetTransform(transform);
	metric->SetInterpolator(interpolator);
	TransformType::ParametersType params(transform->GetNumberOfParameters());
	MetricType::DerivativeType deriv;
	deriv.Fill(0.0);
	params.Fill(0.0);
	metric->Initialize();
	for (double x = -10.0; x <= 10.0; x += 2.0)
	{
		params(0) = x;
		for (double y = -10.0; y <= 10.0; y += 2.0)
		{
			params(1) = y;
			std::cout << params << ": " << metric->GetValue(params) << " ";
			metric->GetDerivative(params, deriv);
			for (unsigned i = 0; i < deriv.GetNumberOfElements(); ++i)
			{
				std::cout << deriv.GetElement(i) << " ";
			}
			std::cout << std::endl;
		}
	}

	return EXIT_SUCCESS;
}