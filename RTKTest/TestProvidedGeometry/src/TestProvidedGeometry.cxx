//
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

#include <itkCudaImage.h>
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkCudaForwardProjectionImageFilter.h"

#include "itkLabelImageToShapeLabelMapFilter.h"
#include "itkRegionOfInterestImageFilter.h"

using VolumePixelType = float;
const unsigned int Dimension = 3;
using CudaVolumeImageType = itk::CudaImage<VolumePixelType, Dimension>;
using CudaVolumeWriterType = itk::ImageFileWriter<CudaVolumeImageType>;

using CudaForwardProjectionFilterType = rtk::CudaForwardProjectionImageFilter<CudaVolumeImageType, CudaVolumeImageType>;
using CudaProjectorPointer =  CudaForwardProjectionFilterType::Pointer;
using RTKProjectionGeometryType = rtk::ThreeDCircularProjectionGeometry;
using RTKGeometryPointer = RTKProjectionGeometryType::Pointer;

using ShapeLabelObjectType = itk::ShapeLabelObject<unsigned short, Dimension>;
using LabelMapType = itk::LabelMap< ShapeLabelObjectType >;
using LabelImageToShapesType = itk::LabelImageToShapeLabelMapFilter<CudaVolumeImageType, LabelMapType>;

using RegionOfInterestFilterType = itk::RegionOfInterestImageFilter<CudaVolumeImageType, CudaVolumeImageType>;


int main(int argc, char *argv[])
{
	// read geometry file
	rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer xmlReader;
	xmlReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
	xmlReader->SetFilename(argv[3]);
	TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlReader->GenerateOutputInformation())
	
	auto geoRead = xmlReader->GetOutputObject();

	// read image file
	auto cudaimagereader = itk::ImageFileReader<CudaVolumeImageType>::New();
	cudaimagereader->SetFileName(argv[1]);
	auto cudaimage = cudaimagereader->GetOutput();

	try
	{
		cudaimage->Update();
	}
	catch (...)
	{
		std::cout << "read intensity error" << std::endl;
	}

	// read label file
	auto labelreader = itk::ImageFileReader<CudaVolumeImageType>::New();
	labelreader->SetFileName(argv[2]);
	auto labelimage = labelreader->GetOutput();

	try
	{
		labelimage->Update();
	}
	catch(...)
	{
		std::cout << "read label error" << std::endl;
	}

	// get label properties
	auto labelimagefilter = LabelImageToShapesType::New();
	labelimagefilter->SetInput(labelimage);
	try
	{
		labelimagefilter->Update();
	}
	catch(...)
	{
		std::cout << "label filter error" << std::endl;
	}
	
	auto regionofinterest = RegionOfInterestFilterType::New();
	regionofinterest->SetInput(labelimage);

	auto regionwriter = CudaVolumeWriterType::New();

	auto labelMap = labelimagefilter->GetOutput();
	std::cout << "File " << "\"" << argv[2] << "\"" << " has " << labelMap->GetNumberOfLabelObjects() << " labels." << std::endl;
	// Retrieve all attributes
	for (unsigned int n = 0; n < labelMap->GetNumberOfLabelObjects(); ++n)
	{
		auto labelObject = labelMap->GetNthLabelObject(n);
		std::cout << "Label: "                            << itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel()) << std::endl;
		std::cout << "    BoundingBox: "                  << labelObject->GetBoundingBox() << std::endl;
		std::cout << "    NumberOfPixels: "	              << labelObject->GetNumberOfPixels() << std::endl;
		std::cout << "    PhysicalSize: "  	    	      << labelObject->GetPhysicalSize() << std::endl;
		std::cout << "    Centroid: "			          << labelObject->GetCentroid() << std::endl;
		std::cout << "    NumberOfPixelsOnBorder: "       << labelObject->GetNumberOfPixelsOnBorder() << std::endl;
		std::cout << "    PerimeterOnBorder: "		      << labelObject->GetPerimeterOnBorder() << std::endl;
		std::cout << "    FeretDiameter: "			      << labelObject->GetFeretDiameter() << std::endl;
		std::cout << "    PrincipalMoments: "		      << labelObject->GetPrincipalMoments() << std::endl;
		std::cout << "    PrincipalAxes: "			      << labelObject->GetPrincipalAxes() << std::endl;
		std::cout << "    Elongation: "  		          << labelObject->GetElongation() << std::endl;
		std::cout << "    Perimeter: "  		          << labelObject->GetPerimeter() << std::endl;
		std::cout << "    Roundness: "  			      << labelObject->GetRoundness() << std::endl;
		std::cout << "    EquivalentSphericalRadius: "    << labelObject->GetEquivalentSphericalRadius() << std::endl;
		std::cout << "    EquivalentSphericalPerimeter: " << labelObject->GetEquivalentSphericalPerimeter() << std::endl;
		std::cout << "    EquivalentEllipsoidDiameter: "  << labelObject->GetEquivalentEllipsoidDiameter() << std::endl;
		std::cout << "    Flatness: "                     << labelObject->GetFlatness() << std::endl;
		std::cout << "    PerimeterOnBorderRatio: "       << labelObject->GetPerimeterOnBorderRatio() << std::endl;

		auto BB = labelObject->GetBoundingBox();
		auto start = BB.GetIndex();
		auto size = BB.GetSize();
		CudaVolumeImageType::RegionType region(start,size);

		regionofinterest->SetRegionOfInterest(region);
		regionwriter->SetInput(regionofinterest->GetOutput());
		char buffer[100];
		sprintf_s(buffer, "Label_%d.nrrd", n);
		regionwriter->SetFileName(buffer);
		try
		{
			regionwriter->Update();
		}
		catch(...)
		{
			std::cout << "region writer error" << std::endl;
		}
			
	}
	
	auto projector = CudaForwardProjectionFilterType::New();
	projector->SetGeometry(geoRead);

	using ProjectionVectorType = std::vector<CudaVolumeImageType::Pointer>;
	ProjectionVectorType projections;

	for (auto i = 0; i < labelMap->GetNumberOfLabelObjects(); ++i)
	{
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
		spacing[0] = 0.2;
		spacing[1] = 0.2;
		spacing[2] = 1;
		projection->SetSpacing(spacing);
		projections.push_back(projection);
	}
	
	auto cudaimagefilewriter = CudaVolumeWriterType::New();
	for (auto i = 0; i < labelMap->GetNumberOfLabelObjects(); ++i)
	{
		auto labelObject = labelMap->GetNthLabelObject(i);
		auto BB = labelObject->GetBoundingBox();
		auto BBstart = BB.GetIndex();
		auto BBsize = BB.GetSize();
		CudaVolumeImageType::RegionType BBregion(BBstart, BBsize);

		regionofinterest->SetRegionOfInterest(BBregion);
		projector->SetInput(1, regionofinterest->GetOutput());
		projector->SetInput(projections[i]);

		char buffer[100];
		sprintf_s(buffer, "ProjLabel_%d.nrrd", i);		
		cudaimagefilewriter->SetFileName(buffer);
		cudaimagefilewriter->SetInput(projector->GetOutput());

		try
		{
			cudaimagefilewriter->Update();
		}
		catch (...)
		{
			std::cout << "writer error" << std::endl;
		}
	}
	
	
	return EXIT_SUCCESS;
}
