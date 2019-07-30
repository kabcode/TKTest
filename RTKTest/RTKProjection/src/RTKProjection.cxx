//
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkEuler3DTransform.h"
#include "itkChangeInformationImageFilter.h"

#include <itkCudaImage.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkCudaForwardProjectionImageFilter.h"

using VolumePixelType = float;
const unsigned int Dimension = 3;
using CudaVolumeImageType = itk::CudaImage<VolumePixelType, Dimension>;
using CudaVolumeWriterType = itk::ImageFileWriter<CudaVolumeImageType>;
using ChangeInformationType = itk::ChangeInformationImageFilter<CudaVolumeImageType>;

using CudaForwardProjectionFilterType = rtk::CudaForwardProjectionImageFilter<CudaVolumeImageType, CudaVolumeImageType>;
using CudaProjectorPointer = CudaForwardProjectionFilterType::Pointer;
using RTKProjectionGeometryType = rtk::ThreeDCircularProjectionGeometry;
using RTKGeometryPointer = RTKProjectionGeometryType::Pointer;


int main(int argc, char *argv[])
{

	std::cout << argv[1] << std::endl;
	std::cout << argv[2] << std::endl;

	// read geometry file
	rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer xmlReader;
	xmlReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
	xmlReader->SetFilename(argv[1]);
	TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlReader->GenerateOutputInformation());

	auto geoRead = xmlReader->GetOutputObject();
	geoRead->Print(std::cout);

	// read image file
	auto cudaimagereader = itk::ImageFileReader<CudaVolumeImageType>::New();
	cudaimagereader->SetFileName(argv[2]);

	try
	{
		cudaimagereader->Update();
	}
	catch (itk::ExceptionObject &EO)
	{
		EO.Print(std::cout);
		std::cout << "read intensity error" << std::endl;
		return EXIT_FAILURE;
	}

	auto projector = CudaForwardProjectionFilterType::New();
	projector->SetGeometry(geoRead);

	auto projection = CudaVolumeImageType::New();
	CudaVolumeImageType::IndexType idx;
	idx[0] = 0;
	idx[1] = 0;
	idx[2] = 0;
	CudaVolumeImageType::SizeType size;
	size[0] = 1200;
	size[1] = 1400;
	size[2] = 1;
	CudaVolumeImageType::RegionType region(idx, size);
	projection->SetRegions(region);
	projection->Allocate();
	projection->FillBuffer(0);

	CudaVolumeImageType::PointType origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	projection->SetOrigin(origin);

	CudaVolumeImageType::SpacingType spacing;
	spacing[0] = 0.15;
	spacing[1] = 0.15;
	spacing[2] = 1;
	projection->SetSpacing(spacing);

	
	projector->SetInput(1, cudaimagereader->GetOutput());
	projector->SetInput(projection);

	auto ChangeInformationFilter = ChangeInformationType::New();
	ChangeInformationFilter->SetInput(projector->GetOutput());
	auto RotationTransform = itk::Euler3DTransform<double>::New();
	RotationTransform->SetComputeZYX(true);
	RotationTransform->SetRotation(
		geoRead->GetOutOfPlaneAngles()[0],
		geoRead->GetInPlaneAngles()[0],
		geoRead->GetGantryAngles()[0]);

	auto R = RotationTransform->GetMatrix();
	auto sid = geoRead->GetSourceToIsocenterDistances();
	auto ssd = geoRead->GetSourceToDetectorDistances();

	CudaVolumeImageType::PointType Origin;
	Origin[0] = -static_cast<double>(size[0]) / 2. * spacing[0];
	Origin[1] = -static_cast<double>(size[1]) / 2. * spacing[1];
	Origin[2] = -(ssd[0] - sid[0]);
	Origin = R * Origin;
	std::cout << Origin << std::endl;
	ChangeInformationFilter->SetOutputOrigin(Origin);
	ChangeInformationFilter->ChangeOriginOn();

	CudaVolumeImageType::DirectionType Direction;
	Direction.Fill(0);
	Direction[0][0] = 1;
	Direction[1][1] = 1;
	Direction[2][2] = 1;
	Direction = R * Direction;
	std::cout << Direction << std::endl;
	ChangeInformationFilter->SetOutputDirection(Direction);
	ChangeInformationFilter->ChangeDirectionOn();

	auto cudaimagefilewriter = CudaVolumeWriterType::New();
	cudaimagefilewriter->SetFileName("ProjLabel.nrrd");
	cudaimagefilewriter->SetInput(ChangeInformationFilter->GetOutput());

	try
	{
		cudaimagefilewriter->Update();
	}
	catch (...)
	{
		std::cout << "writer error" << std::endl;
	}


	return EXIT_SUCCESS;
}
