// ITK includes
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkGradientImageFilter.h"
#include "HighPrecisionTimer.h"

#include <execution>
#include "itkImageBufferRange.h"

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

	auto binary_op = [](VectorPixel num1, VectorPixel num2) {
		VectorPixel sum;
		for (auto i = 0; i < VectorPixel::Dimension; ++i)
			sum[i] = num1[i] + num2[i];
		return sum; };

	auto img = VectorImageType::New();

	VectorImageType::IndexType idx;
	idx.Fill(0);

	VectorImageType::SizeType sz;
	sz.Fill(512);
	sz[2] = 1;

	VectorImageType::RegionType reg(idx, sz);
	img->SetRegions(reg);
	img->Allocate();
	VectorPixel px;
	px[0] = 0.1;
	px[1] = 0.2;
	px[2] = 0.3;
	img->FillBuffer(px);

	for (auto i = 0; i < 4; ++i)
	{
		itk::ImageRegionConstIterator<VectorImageType> iter(img, img->GetLargestPossibleRegion());
		iter.GoToBegin();
		VectorPixel sum;
		sum.Fill(0);

		{
			auto Timer{ HighPrecisionTimer<TimeUnits::Microseconds>() };
			while (!iter.IsAtEnd())
			{
				sum = sum + iter.Get();
				++iter;
			}
		}
		if(i == 3) std::cout << sum << std::endl;

		itk::Experimental::ImageBufferRange<VectorImageType> range{ *img };
		sum.Fill(0);
		{
			auto Timer{ HighPrecisionTimer<TimeUnits::Microseconds>() };
			for (auto&& var : range)
			{
				sum = sum + var;
			}
		}
		if (i == 3)  std::cout << sum << std::endl;

		sum.Fill(0);
		{
			auto Timer{ HighPrecisionTimer<TimeUnits::Microseconds>() };
			sum = std::reduce(std::execution::seq, range.begin(), range.end(), sum);
		}
		if (i == 3)  std::cout << sum << std::endl;

		sum.Fill(0);
		{
			auto Timer{ HighPrecisionTimer<TimeUnits::Microseconds>() };
			sum = std::reduce(std::execution::par, range.begin(), range.end(), sum);
		}
		if (i == 3)  std::cout << sum << std::endl;

		sum.Fill(0);
		{
			auto Timer{ HighPrecisionTimer<TimeUnits::Microseconds>() };
			sum = std::reduce(std::execution::par_unseq, range.begin(), range.end(), sum);
		}
		if (i == 3)  std::cout << sum << std::endl;
	}

	/*
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
	EulerTransformType::OutputVectorType translation;
	translation.Fill(15);
	translation[2] = 0;
	Transform->SetTranslation(translation);

	auto ResampleFilter = ResampleVectorImageFilterType::New();
	ResampleFilter->SetInput(GradientImageFilter->GetOutput());
	ResampleFilter->SetTransform(Transform);
	ResampleFilter->SetDefaultPixelValue(itk::NumericTraits<VectorPixel>::ZeroValue());

	ResampleFilter->SetSize(ImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
	ResampleFilter->SetOutputSpacing(ImageReader->GetOutput()->GetSpacing());

	//const auto OutputOrigin = Transform->GetInverseTransform()->TransformPoint(ImageReader->GetOutput()->GetOrigin());
	//ResampleFilter->SetOutputOrigin(OutputOrigin);
	ResampleFilter->SetOutputOrigin(ImageReader->GetOutput()->GetOrigin());
	const auto OutputDirection = Transform->GetMatrix().GetInverse() * ImageReader->GetOutput()->GetDirection().GetVnlMatrix();
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

	const auto sum = std::accumulate(durations.begin(), durations.end(), 0LL);
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
	*/
	return EXIT_SUCCESS;

}
