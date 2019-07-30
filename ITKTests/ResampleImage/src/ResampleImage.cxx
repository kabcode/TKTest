// STL includes
#include <filesystem>

// ITK includes
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"
#include "itkEuler3DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkExtractImageFilter.h"


using PixelType = float;
const unsigned int Dim3D = 3;
const unsigned int Dim2D = 2;
using InputImageType = itk::Image<PixelType, Dim3D>;
using OutputImageType = itk::Image<PixelType, Dim3D>;

using ImageReaderType = itk::ImageFileReader<InputImageType>;
using ImageWriterType = itk::ImageFileWriter<OutputImageType>;
using TransformReaderType = itk::TransformFileReaderTemplate<double>;
using TransformWriterType = itk::TransformFileWriterTemplate<double>;
using OutputImageWriterType = itk::ImageFileWriter<OutputImageType>;
using EulerTransformType = itk::Euler3DTransform<double>;
using EulerTransformPointer = EulerTransformType::Pointer;
using ResampleImageFilterType = itk::ResampleImageFilter<InputImageType, OutputImageType>;

double inline deg2rad(double deg) { return deg / 180 * (itk::Math::pi / 2); }

void
PrintUsage(char * programmname)
{
	std::cout << "\nInformation to " << programmname << std::endl;
	std::cout << "Usage: " << std::endl;
	std::cout << programmname << " VolumeFile -o OutputFile <optional parameters>" << std::endl;
	std::cout << "\noptional parameters:\n" << std::endl;
	std::cout << "-t x y z \t\t Translation in x,y,z direction in mm" << std::endl;
	std::cout << "-r x y z \t\t Rotation around x,y,z axis in degree(order: ZXY)" << std::endl;
	std::cout << "-f Transformfilename \t\t File with transformation parameter" << std::endl;
	std::cout << std::endl;
}

int main(int argc, char* argv[])
{
	// check input arguments
	if (argc < 3)
	{
		PrintUsage(argv[0]);
		return EXIT_FAILURE;
	}
	std::string OutputFilename("");
	std::string TransformationFileName("");

	// input handling
	std::vector<double> Translation = { 0,0,0 };
	std::vector<double> Rotation = { 0,0,0 };
	for (auto i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-o")
		{
			++i;
			OutputFilename = argv[i];
		}
		if (std::string(argv[i]) == "-f")
		{
			++i;
			TransformationFileName = argv[i];
		}
		if (std::string(argv[i]) == "-t")
		{
			++i;
			Translation[0] = std::atof(argv[i]);
			++i;
			Translation[1] = std::atof(argv[i]);
			++i;
			Translation[2] = std::atof(argv[i]);
		}
		if (std::string(argv[i]) == "-r")
		{
			++i;
			Rotation[0] = std::atof(argv[i]);
			++i;
			Rotation[1] = std::atof(argv[i]);
			++i;
			Rotation[2] = std::atof(argv[i]);
		}
	}

	auto ImageReader = ImageReaderType::New();
	ImageReader->SetFileName(argv[1]);
	try
	{
		ImageReader->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}


	auto T = EulerTransformType::New();
	if (TransformationFileName.empty())
	{
		T->SetRotation(deg2rad(Rotation[0]), deg2rad(Rotation[1]), deg2rad(Rotation[2]));
		EulerTransformType::OutputVectorType TranslationVector;
		for (auto i = 0; i < Translation.size(); ++i)
		{
			TranslationVector[i] = Translation[i];
		}
		T->SetTranslation(TranslationVector);
	}
	else
	{
		auto TransformReader = itk::TransformFileReader::New();
		TransformReader->SetFileName(TransformationFileName);
		TransformReader->Update();

		const TransformReaderType::TransformListType * transforms = TransformReader->GetTransformList();
		auto it = transforms->begin();
		if (!strcmp((*it)->GetNameOfClass(), "Euler3DTransform"))
		{
			T = static_cast< EulerTransformType* >((*it).GetPointer());
		}
		T.Print(std::cout);
	}
	
	auto TransformWriter = itk::TransformFileWriterTemplate<double>::New();
	TransformWriter->SetInput(T);
	TransformWriter->SetFileName("Transform.tfm");
	TransformWriter->Update();

	auto ResampleFilter = ResampleImageFilterType::New();
	ResampleFilter->SetInput(ImageReader->GetOutput());
	ResampleFilter->SetTransform(T);
	ResampleFilter->SetDefaultPixelValue(0);

	ResampleFilter->SetSize(ImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
	ResampleFilter->SetOutputSpacing(ImageReader->GetOutput()->GetSpacing());

	auto OutputOrigin = T->GetInverseTransform()->TransformPoint(ImageReader->GetOutput()->GetOrigin());
	ResampleFilter->SetOutputOrigin(OutputOrigin);
	auto OutputDirection = T->GetInverseMatrix() * ImageReader->GetOutput()->GetDirection();
	ResampleFilter->SetOutputDirection(OutputDirection);

	try
	{
		ResampleFilter->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}

	auto ImageWriter = ImageWriterType::New();
	ImageWriter->SetInput(ResampleFilter->GetOutput());
	ImageWriter->SetFileName(OutputFilename);

	try
	{
		ImageWriter->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}
