#include "itkCudaImage.h"
#include "itkEuler3DTransform.h"
#include "itkAmoebaOptimizer.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkGaussianInterpolateImageFunction.h"
#include "itkInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

using ImageType = itk::CudaImage<float, 3>;


int main(int argc, char* argv[])
{

	std::cout << "\n========================================" << std::endl;
	std::cout << "\n          Transform Clone Test          " << std::endl;
	std::cout << "========================================\n" << std::endl;

	using TransformType = itk::Euler3DTransform<double>;

	// Create master transform
	auto Transform = TransformType::New();
	auto Parameters = Transform->GetParameters();
	for (auto i = 0; i < Parameters.GetSize(); ++i)
	{
		Parameters[i] = i;
	}

	Transform->SetParameters(Parameters);
	Transform->Print(std::cout);
	std::cout << "========================================" << std::endl;
	// Create another transform
	auto AnotherTransform = Transform->CreateAnother();
	AnotherTransform->Print(std::cout);
	std::cout << "========================================" << std::endl;
	// Clone transform
	auto ClonedTransform = Transform->Clone();
	ClonedTransform->Print(std::cout);

	std::cout << "\n========================================" << std::endl;
	std::cout << "\n          Optimizer Clone Test          " << std::endl;
	std::cout << "========================================\n" << std::endl;

	using OptimizerType = itk::AmoebaOptimizer;

	auto Optimizer = OptimizerType::New();
	Optimizer->SetMaximumNumberOfIterations(321);
	Optimizer->SetMaximize(true);
	Optimizer->SetFunctionConvergenceTolerance(0.12);
	Optimizer->SetOptimizeWithRestarts(false);
	OptimizerType::ScalesType scales(Transform->GetNumberOfParameters());
	scales[0] = 10;
	scales[1] = 20;
	scales[2] = 30;
	scales[3] = 1010;
	scales[4] = 1020;
	scales[5] = 1030;
	Optimizer->SetScales(scales);
	Optimizer->Print(std::cout);
	std::cout << "========================================" << std::endl;

	auto AnotherOptimizer = Optimizer->CreateAnother();
	AnotherOptimizer->Print(std::cout);
	std::cout << "========================================" << std::endl;

	auto ClonedOptimizer = Optimizer->Clone();
	ClonedOptimizer->Print(std::cout);
	std::cout << "========================================" << std::endl;

	std::cout << "\n========================================" << std::endl;
	std::cout << "\n            Metric Clone Test           " << std::endl;
	std::cout << "========================================\n" << std::endl;

    using MetricType = itk::ImageToImageMetric<ImageType, ImageType>;
    using MetricTypePointer = MetricType::Pointer;
    auto VecMetric = std::vector<MetricTypePointer>();

	using MSMetricType = itk::MeanSquaresImageToImageMetric<ImageType, ImageType>;

	auto MSMetric = MSMetricType::New();
    MSMetric->SetUseAllPixels(false);
    MSMetric->SetUseFixedImageIndexes(true);
    MSMetric->SetComputeGradient(false);
    VecMetric.push_back(MSMetric);
    MSMetric.Print(std::cout);
	std::cout << "========================================" << std::endl;

	auto AnotherMetric = MSMetric->CreateAnother();
	AnotherMetric->Print(std::cout);
	std::cout << "========================================" << std::endl;

	const auto ClonedMetric = MSMetric->Clone();
    VecMetric.push_back(ClonedMetric);
	ClonedMetric->Print(std::cout);
	std::cout << "========================================" << std::endl;

	std::cout << "\n========================================" << std::endl;
	std::cout << "\n          Interpolator Clone Test          " << std::endl;
	std::cout << "========================================\n" << std::endl;

	using InterpolatorType = itk::InterpolateImageFunction<ImageType, typename itk::NumericTraits< typename ImageType::PixelType >::RealType>;
	using InterpolatorTypePointer = InterpolatorType::Pointer;
	auto VecInterp = std::vector<InterpolatorTypePointer>();

	const auto NNInterpolator = itk::NearestNeighborInterpolateImageFunction<ImageType>::New();
	const InterpolatorTypePointer masterinterp = NNInterpolator;

	const auto GInterpolator = itk::GaussianInterpolateImageFunction<ImageType>::New();
	VecInterp.push_back(GInterpolator);
	VecInterp[0]->Print(std::cout);

	const auto LInterpolator = itk::LinearInterpolateImageFunction<ImageType>::New();
	VecInterp.push_back(LInterpolator);
	VecInterp[1]->Print(std::cout);
	std::cout << "========================================" << std::endl;

	auto AnotherInterpolator = GInterpolator->Clone();
	const auto image = ImageType::New();
	VecInterp.push_back(AnotherInterpolator);
	AnotherInterpolator->SetInputImage(image);
	VecInterp[2]->Print(std::cout);
	std::cout << "========================================" << std::endl;

	const auto ClonedInterpolator = NNInterpolator->Clone();
	ClonedInterpolator->Print(std::cout);
	VecInterp.push_back(ClonedInterpolator);
	VecInterp[3]->Print(std::cout);

    std::cout << "========================================" << std::endl;
    const auto Interpolator = InterpolatorType::New();
    auto CloneInterpolator = Interpolator->Clone();
    CloneInterpolator->Print(std::cout);


	return EXIT_SUCCESS;
}