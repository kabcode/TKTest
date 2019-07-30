
#include "itkImage.h"
#include "itkImageFileWriter.h"

#include "itkResampleImageFilter.h"
#include "itkEuler3DTransform.h"

typedef itk::Image<float, 3> ImageType;
typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilterType;
typedef itk::Euler3DTransform<double> EulerTransformType;


int main()
{
	auto image = ImageType::New();

	ImageType::RegionType region;
	ImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;
	start[2] = 0;

	ImageType::SizeType size;
	size[0] = 512;
	size[1] = 512;
	size[2] = 1;

	region.SetSize(size);
	region.SetIndex(start);
	image->SetRegions(region);
	image->Allocate();
	image->FillBuffer(100);

	ImageType::SpacingType sp;
	sp[0] = 1;
	sp[1] = 1;
	sp[2] = 1;
	image->SetSpacing(sp);

	ImageType::PointType or ;
	or[0] = 0;
	or[1] = 0;
	or[2] = 0;
	image->SetOrigin(or);

	for (auto i = 10; i < 500; ++i)
	{
		for (auto j = 10; j < 500; ++j)
		{
			ImageType::IndexType idx;
			idx[0] = i;
			idx[1] = j;
			idx[2] = 0;
			image->SetPixel(idx, i+j);
		}
	}

	image->Update();
	image->Print(std::cout);

	auto transform = EulerTransformType::New();
	transform->SetRotation(0.1, 0.2, 0);
	transform->Print(std::cout);

	std::cout << "Here" << std::endl;

	auto Resampler = ResampleFilterType::New();
	ImageType::SizeType sz;
	sz[0] = 256;
	sz[1] = 256;
	sz[2] = 1;
	ImageType::PointType ori;
	ori[0] = -100;
	ori[1] = -100;
	ori[2] = -100;
	Resampler->SetInput(image);
	Resampler->SetOutputOrigin(image->GetOrigin());
	//Resampler->SetOutputOrigin(ori);
	Resampler->SetOutputSpacing(image->GetSpacing());
	//Resampler->SetSize(sz);
	ImageType::DirectionType invTransform = transform->GetMatrix().GetInverse();
	Resampler->SetSize(image->GetLargestPossibleRegion().GetSize());
	Resampler->SetOutputDirection(invTransform);
	Resampler->SetTransform(transform);
	Resampler->SetDefaultPixelValue(150);

	try
	{
		Resampler->Update();
	}
	catch(itk::ExceptionObject EO)
	{
		std::cout << "Exception caught!" << std::endl;
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}

	
	Resampler->GetOutput()->Print(std::cout);

	auto writer = itk::ImageFileWriter<ImageType>::New();
	writer->SetInput(Resampler->GetOutput());
	writer->SetFileName("Slice.nrrd");
	writer->Update();

}