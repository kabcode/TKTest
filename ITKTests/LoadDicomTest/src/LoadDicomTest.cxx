#include "itkImageFileReader.h"
#include <cstdlib>

using PixelType = float;
static constexpr unsigned int Dim = 3;
using ImageType = itk::Image<PixelType, Dim>;

using ReaderType = itk::ImageFileReader<ImageType>;


int main(int argc, char* argv[])
{
	std::string ImageFileName = argv[1];

	auto Reader = ReaderType::New();
	Reader->SetFileName(ImageFileName);

	try
	{
		Reader->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
	}

	auto Image = Reader->GetOutput();

	Image->Print(std::cout);



	return EXIT_SUCCESS;
}
