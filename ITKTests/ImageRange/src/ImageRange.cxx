// STL includes
#include <filesystem>

// ITK includes
#include "itkImageFileReader.h"
#include "itkEuler3DTransform.h"

using PixelType = float;
const unsigned int Dim3D = 3;
using ImageType = itk::Image<PixelType, Dim3D>;

using ImageReaderType = itk::ImageFileReader<ImageType>;




int main(int argc, char* argv[])
{
	/*
	auto ImageReader = ImageReaderType::New();
	ImageReader->SetFileName(argv[1]);
	ImageReader->Update();

	auto Image = ImageReader->GetOutput();

	auto si = Image->GetLargestPossibleRegion().GetSize();
	auto sp =Image->GetSpacing();
	auto or = Image->GetOrigin();
	auto di = Image->GetDirection();

	itk::Vector<double> rangeX;
	rangeX.Fill(0);
	rangeX[0] = si[0] * sp[0];
	rangeX = di * rangeX;
	itk::Vector<double> rangeY;
	rangeY.Fill(0);
	rangeY[1] = si[1] * sp[1];
	rangeY = di * rangeY;
	itk::Vector<double> rangeZ;
	rangeZ.Fill(0);
	rangeZ[2] = si[2] * sp[2];
	rangeZ = di * rangeZ;
	*/

	std::vector<itk::Euler3DTransform<double>::Pointer> vec;
	
	for (auto i = 0; i < 3; ++i)
	{
		auto t = itk::Euler3DTransform<double>::New();
		vec.push_back(t);
	}

	vec.resize(10);
	std::cout << vec.capacity() << std::endl;
	std::cout << vec.size() << std::endl;

	vec.emplace_back(itk::Euler3DTransform<double>::New());
	vec.emplace_back(itk::Euler3DTransform<double>::New());
	vec.emplace_back(itk::Euler3DTransform<double>::New());

	for (auto &i : vec)
		std::cout << i.GetPointer() << std::endl;

	std::cout << vec.capacity() << std::endl;
	std::cout << vec.size() << std::endl;

	vec.shrink_to_fit();

	std::cout << vec.capacity() << std::endl;
	std::cout << vec.size() << std::endl;

	vec.resize(vec.size());

	std::cout << vec.capacity() << std::endl;
	std::cout << vec.size() << std::endl;

	for (auto &i : vec)
		std::cout << i.GetPointer() << std::endl;

	return EXIT_SUCCESS;

}
