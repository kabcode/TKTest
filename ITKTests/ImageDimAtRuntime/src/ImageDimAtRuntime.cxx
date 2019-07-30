// STL includes
#include <filesystem>

// ITK includes
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

using PixelType = float;
const unsigned int Dim3D = 3;
using Image3DType = itk::Image<PixelType, Dim3D>;

const unsigned int Dim2D = 2;
using Image2DType = itk::Image<PixelType, Dim2D>;

using Image3DReaderType = itk::ImageFileReader<Image3DType>;
using Image2DReaderType = itk::ImageFileReader<Image2DType>;


Image3DType::Pointer Make3DFrom2DImage(Image2DType::Pointer image);


int main(int argc, char* argv[])
{
	auto Image = Image3DType::New();

	auto Image3DReader = Image3DReaderType::New();
	Image3DReader->SetFileName(argv[1]);
	try
	{
		Image3DReader->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
	}

	Image = Image3DReader->GetOutput();
	Image.Print(std::cout);

	auto ImageWriter = itk::ImageFileWriter<Image3DType>::New();
	ImageWriter->SetInput(Image);
	ImageWriter->SetFileName("Output.nrrd");
	ImageWriter->Update();

	
	return EXIT_SUCCESS;

}

Image3DType::Pointer
Make3DFrom2DImage(Image2DType::Pointer image)
{
	return nullptr;
}
