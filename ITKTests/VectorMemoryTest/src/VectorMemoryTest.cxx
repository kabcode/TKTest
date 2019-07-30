#include "VectorKernel.hcu"

// ITK includes
#include "itkCudaImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"

using PixelType = itk::CovariantVector<float, 3>;
using ImageType = itk::CudaImage<PixelType, 3>;


int main(int argc, char* argv[])
{
	auto Image = ImageType::New();
	ImageType::SizeType size;
	size[0] = 3;
	size[1] = 4;
	size[2] = 5;
	ImageType::IndexType index;
	index.Fill(0);
	ImageType::RegionType region(index, size);

	Image->SetRegions(region);
	Image->Allocate();
	PixelType v;
	v[0] = 11;
	v[1] = 22;
	v[2] = 33;
	Image->FillBuffer(v);

	itk::ImageRegionIterator<ImageType> it(Image, Image->GetLargestPossibleRegion());

	int sizeVec[] = { Image->GetLargestPossibleRegion().GetSize()[0],Image->GetLargestPossibleRegion().GetSize()[1], Image->GetLargestPossibleRegion().GetSize()[2] };
	int lengthVec = PixelType::Length;
	float* vec = *(float**) Image->GetCudaDataManager()->GetGPUBufferPointer();
	CUDA_writeVector(vec, sizeVec, lengthVec);
	cudaDeviceSynchronize();

	auto cpubuffer = Image->GetCudaDataManager()->GetCPUBufferPointer();
	auto sizeX = Image->GetLargestPossibleRegion().GetSize()[0];
	auto sizeY = Image->GetLargestPossibleRegion().GetSize()[1];
	auto sizeZ = Image->GetLargestPossibleRegion().GetSize()[2];
	auto sizeV = PixelType::Length;

	for (auto k = 0; k < sizeZ; ++k)
	{
		for (auto j = 0; j < sizeY; ++j)
		{
			for (auto i = 0; i < sizeX; ++i)
			{
				for (auto c = 0; c < PixelType::Length; ++c)
					std::cout << static_cast<float*>(cpubuffer)[c + sizeV * (i + sizeX * (j + k * sizeY))] << std::endl;
			}
		}
	}

	auto writer = itk::ImageFileWriter<itk::CudaImage<itk::CovariantVector<float, 3>, 3>>::New();
	writer->SetInput(Image);
	writer->SetFileName("workaround.nrrd");
	writer->Update();

	return EXIT_SUCCESS;

}
