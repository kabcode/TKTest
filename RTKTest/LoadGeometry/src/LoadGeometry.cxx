#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"


using PixelType = float;
static constexpr unsigned int Dim2 = 2;
static constexpr unsigned int Dim3 = 3;
using GeometryType = rtk::ThreeDCircularProjectionGeometry;
using GeometryPointer = GeometryType::Pointer;
using GeometryReaderType = rtk::ThreeDCircularProjectionGeometryXMLFileReader;



int main(int argc, char* argv[])
{
	
	// Read Projection geometry file
	auto GeometryReader = GeometryReaderType::New();
	std::cout << GeometryReader->CanReadFile(argv[1]) << std::endl;
	std::cout << "File: " << argv[1] << std::endl;
	GeometryReader->SetFilename(argv[1]);
	GeometryReader->GenerateOutputInformation();
	auto Geometry = GeometryReader->GetOutputObject();
	Geometry->Print(std::cout);

	auto Gem = Geometry;

	std::cout << Gem->GetGantryAngles()[0] << std::endl;

	return EXIT_SUCCESS;
}