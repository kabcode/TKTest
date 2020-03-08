#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkTranslationTransform.h"

int main(int argc, char* argv[])
{
	using ImageType = itk::Image< double, 2 >;
	using ReaderType = itk::ImageFileReader<ImageType>;
	using InterpolatorType = itk::NearestNeighborInterpolateImageFunction<ImageType, double >;
	using TransformType = itk::TranslationTransform < double, 2 >;

	using MetricType = itk::MeanSquaresImageToImageMetric < ImageType, ImageType >;
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
	
	
	MetricType::Pointer metric = MetricType::New();
	TransformType::Pointer transform = TransformType::New();
	transform->SetIdentity();
	auto parameters = transform->GetParameters();

	InterpolatorType::Pointer interpolator = InterpolatorType::New();

	metric->SetFixedImage(fixedImage);
	metric->SetMovingImage(movingImage);
	metric->SetFixedImageRegion(fixedImage->GetLargestPossibleRegion());
	metric->SetTransform(transform);
	metric->SetInterpolator(interpolator);
	metric->Initialize();

	const auto MSE = metric->GetValue(parameters);

	const auto PSNR = 20 * std::log10(itk::NumericTraits<double>::max() / std::sqrt(MSE));
	std::cout << "PSNR:" << PSNR << std::endl;

	return EXIT_SUCCESS;
}
