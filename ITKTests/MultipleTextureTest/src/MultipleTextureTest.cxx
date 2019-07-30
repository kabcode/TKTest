#include "VectorKernel.hcu"

// ITK includes
#include "itkCudaImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include <chrono>
#include <random>
#include <thrust/sort.h>

using PixelType = itk::CovariantVector<float, 3>;
using ImageType = itk::CudaImage<PixelType, 3>;


int main(int argc, char* argv[])
{
	auto GradientFileReader = itk::ImageFileReader<ImageType>::New();
	GradientFileReader->SetFileName(argv[1]);
	auto CovImage = GradientFileReader->GetOutput();
	CovImage->Update();
	auto pin = *(float**)(CovImage->GetCudaDataManager()->GetGPUBufferPointer());
	auto sz = CovImage->GetLargestPossibleRegion().GetSize();
	int size[] = { sz[0], sz[1], sz[2] };

	auto GradientMagnitudeImage = itk::CudaImage<float, 3>::New();
	itk::CudaImage<float, 3>::IndexType idx;
	idx.Fill(0);
	itk::CudaImage<float, 3>::RegionType reg(idx, sz);
	GradientMagnitudeImage->SetRegions(reg);
	GradientMagnitudeImage->Allocate();
	GradientMagnitudeImage->FillBuffer(0);
	auto pout = *(float**)(GradientMagnitudeImage->GetCudaDataManager()->GetGPUBufferPointer());

	std::random_device rand_dev;
	std::mt19937 generator(rand_dev());

	const long s = 3 * 10000;
	int pts[s];
	std::uniform_int_distribution<int>  distrX(0, size[0]-1);
	std::uniform_int_distribution<int>  distrY(0, size[1]-1);
	std::uniform_int_distribution<int>  distrZ(0, size[2]-1);
	for (auto i = 0; i < s; i+=3)
	{
		pts[i + 0] = distrX(generator);
		pts[i + 1] = distrY(generator);
		pts[i + 2] = distrZ(generator);
	}

	using us = std::chrono::microseconds;
	
	auto startMT = std::chrono::high_resolution_clock::now();
	CUDA_useMultipleTextures(pin,pout,size,3, pts);
	auto stopMT = std::chrono::high_resolution_clock::now();
	auto durMT(stopMT - startMT);
	std::cout << "Multiple textures: "<< std::chrono::duration_cast<us>(durMT).count()  << " us" << std::endl;

	auto MTFileWriter = itk::ImageFileWriter<itk::CudaImage<float, 3>>::New();
	MTFileWriter->SetFileName("GradMT.nrrd");
	MTFileWriter->SetInput(GradientMagnitudeImage);
	MTFileWriter->Update();
	
	auto startST = std::chrono::high_resolution_clock::now();
	//CUDA_useSingleTexture(pin, pout, size, 3);
	auto stopST = std::chrono::high_resolution_clock::now();
	auto durST(stopST - startST);
	std::cout << "Single texture: " << std::chrono::duration_cast<us>(durST).count() << " us" << std::endl;

	auto STFileWriter = itk::ImageFileWriter<itk::CudaImage<float, 3>>::New();
	STFileWriter->SetFileName("GradST.nrrd");
	STFileWriter->SetInput(GradientMagnitudeImage);
	STFileWriter->Update();
	
	auto startNT = std::chrono::high_resolution_clock::now();
	CUDA_useNoTexture(pin, pout, size, 3);
	auto stopNT = std::chrono::high_resolution_clock::now();
	auto durNT(stopNT - startNT);
	std::cout << "No textures: " << std::chrono::duration_cast<us>(durNT).count() << " us" << std::endl;

	auto NTFileWriter = itk::ImageFileWriter<itk::CudaImage<float, 3>>::New();
	NTFileWriter->SetFileName("GradNT.nrrd");
	NTFileWriter->SetInput(GradientMagnitudeImage);
	NTFileWriter->Update();
	CovImage = nullptr;

	return EXIT_SUCCESS;

}
