#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include <itksys/SystemTools.hxx>

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
typedef rtk::ThreeDCircularProjectionGeometryXMLFileWriter GeometryFileWriterType;
typedef rtk::ThreeDCircularProjectionGeometryXMLFileReader GeometryFileReaderType;

int main(int argc, char* argv[]  )
{
	if(argc < 3)
	{
		std::cout << argv[0] << " numberofcircularprojections GeometryOutputFile" << std::endl;
		return EXIT_FAILURE;
	}


    // Create a geometry object with n projection
	auto geometry = GeometryType::New();

	unsigned int firstAngle = 0;
	unsigned int angularArc = 360;
	double isox = 0;
	double isoy = 0;
	double gantryAngle = 0;
	double outOfPlaneAngle = 0; //rotate around z axis
	double inPlaneAngle = 0; //rotate around z axis
	
	unsigned int numberOfProjections = std::stoi(argv[1]);
	for (auto noProj = 0; noProj < numberOfProjections; ++noProj)
	{
		gantryAngle = (float)firstAngle + (float)noProj * angularArc / (float)numberOfProjections;
		geometry->AddProjection(1000,1536, gantryAngle, isox, isoy, outOfPlaneAngle, inPlaneAngle);
	}
	

	std::string GeometryOutputFile(argv[2]);
	auto xmlWriter = GeometryFileWriterType::New();
	xmlWriter->SetFilename(GeometryOutputFile);
	xmlWriter->SetObject(geometry);
	TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlWriter->WriteFile());

	return EXIT_SUCCESS;
}