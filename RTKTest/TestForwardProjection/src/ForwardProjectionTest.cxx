
#include <itkCudaImage.h>
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkTranslationTransform.h>
#include <itkChangeInformationImageFilter.h>
#include <itkEuler3DTransform.h>
#include <itkResampleImageFilter.h>

#include <itkThresholdImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkChangeInformationImageFilter.h>

#include <rtkThreeDCircularProjectionGeometry.h>
#include <rtkCudaForwardProjectionImageFilter.h>
//#include <rtkConstantImageSource.h>

#include <fstream>
#include <regex>
#include <experimental/filesystem>

const unsigned int Dim2D = 2;
const unsigned int Dim3D = 3;
typedef float PixelType;

typedef itk::Image<PixelType, Dim2D> ImageType;
typedef itk::Image<PixelType, Dim3D> VolumeType;
typedef itk::CudaImage<PixelType, Dim3D> CImageType;

typedef itk::ImageSeriesReader<VolumeType> ImageSeriesReaderType;
typedef itk::ImageFileReader< ImageType > ProjectionImageReaderType;
typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;

typedef itk::ThresholdImageFilter<VolumeType> ThresholdImageFilterType;
typedef itk::MultiplyImageFilter<VolumeType> MultiplyImageFilterType;
typedef itk::MultiplyImageFilter<CImageType> MultiplyCImageFilterType;
typedef itk::AddImageFilter<VolumeType> AddImageFilterType;
typedef itk::ChangeInformationImageFilter<VolumeType> ChangeInformationFilterType;

typedef itk::Euler3DTransform<double> EulerTransformType;
typedef itk::ResampleImageFilter<VolumeType, VolumeType> ResamplerFilterType;

typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
typedef rtk::CudaForwardProjectionImageFilter<CImageType, CImageType> CudaForwardProjectorType;

typedef struct
{
	GeometryType::VectorType m_u;
	GeometryType::VectorType m_v;
	GeometryType::PointType m_det;
	GeometryType::PointType m_src;
} DetectorGeometry;

// Function declarations
std::string            LoadDICOMVolume(char* Directory);
std::string			   LoadNRRDVolume(char* FileName);
CImageType::Pointer    LoadCUDAImage(std::string NRRDImageName);
GeometryType::Pointer  LoadProjectionGeometry(char* ProjectionImageFile, DetectorGeometry& ProjectionGeometry);
CImageType::Pointer    SetupCudaProjectionImage(char* ProjectionImageFile);

inline float DegreesToRadians(float degrees) { return degrees * itk::Math::pi / 180; };

int main(int argc, char* argv[])
{
	// input check correct number of input arguments
	if (argc != 4)
	{
		return EXIT_FAILURE;
	}

	// Read DICOM image to NRRD image and write image to disk (Issue with ChangeInformationFilter)
	auto isCentered = false;
	std::string NRRDImageName;
	if(!isCentered)
	{
		std::string extension("");
		std::experimental::filesystem::path VolumePath(argv[1]);
		if (VolumePath.has_extension())
		{
			extension = VolumePath.extension().string();
		}
		if (extension.compare(".nrrd") == 0)
		{
			NRRDImageName = LoadNRRDVolume(argv[1]);
		}
		else
		{
			NRRDImageName = LoadDICOMVolume(argv[1]);
		}
		if (NRRDImageName.compare("error") == 0) return EXIT_FAILURE;
	}
	else
	{
		NRRDImageName = "CenteredVolume.nrrd";
	}

	// Read centered NRRD image from disk to CUDA image
	auto CudaVolume = LoadCUDAImage(NRRDImageName);
	if (CudaVolume == ITK_NULLPTR) return EXIT_FAILURE;
	
	// Create projection geometry from projection image
	DetectorGeometry ProjectionGeometry;
	auto CudaProjectionGeometry = LoadProjectionGeometry(argv[2], ProjectionGeometry);
	if (CudaProjectionGeometry == ITK_NULLPTR) return EXIT_FAILURE;

	// Define Projection image
	auto CudaProjectionImage = SetupCudaProjectionImage(argv[2]);
	if (CudaProjectionImage == ITK_NULLPTR) return EXIT_FAILURE;
	
	auto CudaForwardProjector = CudaForwardProjectorType::New();
	CudaForwardProjector->InPlaceOff();
	CudaForwardProjector->SetInput(0, CudaProjectionImage);
	CudaForwardProjector->SetInput(1, CudaVolume);
	CudaForwardProjector->SetGeometry(CudaProjectionGeometry);

	auto StretchCImageValueRange = MultiplyCImageFilterType::New();
	StretchCImageValueRange->SetInput(CudaForwardProjector->GetOutput());
	StretchCImageValueRange->SetConstant(100);
	StretchCImageValueRange->GetOutput()->UpdateBuffers();
	
	// implicit conversion from CImageType to VolumeType
	VolumeType::Pointer newImage = StretchCImageValueRange->GetOutput();
	newImage->Update();

	auto Changer = ChangeInformationFilterType::New();
	Changer->SetInput(newImage);

	VolumeType::SpacingType sp;
	sp[0] = 0.154;
	sp[1] = 0.154;
	sp[2] = 1;
	Changer->SetOutputSpacing(sp);
	Changer->ChangeSpacingOn();
	
	Changer->SetOutputOrigin(ProjectionGeometry.m_det);
	Changer->ChangeOriginOn();

	auto cross = CrossProduct(ProjectionGeometry.m_u, ProjectionGeometry.m_v);
	std::cout << "cross: " << cross << std::endl;
	VolumeType::DirectionType dir;
	dir[0][0] = ProjectionGeometry.m_u[0];
	dir[1][0] = ProjectionGeometry.m_u[1];
	dir[2][0] = ProjectionGeometry.m_u[2];
	dir[0][1] = ProjectionGeometry.m_v[0];
	dir[1][1] = ProjectionGeometry.m_v[1];
	dir[2][1] = ProjectionGeometry.m_v[2];
	dir[0][2] = cross[0];
	dir[1][2] = cross[1];
	dir[2][2] = cross[2];
	Changer->SetOutputDirection(dir);
	Changer->ChangeDirectionOn();

	// Write Image
	auto ImageWriter = itk::ImageFileWriter<VolumeType>::New();
	ImageWriter->SetInput(StretchCImageValueRange->GetOutput());
	ImageWriter->SetFileName(argv[3]);

	auto ImageWriter2 = itk::ImageFileWriter<VolumeType>::New();
	ImageWriter2->SetInput(Changer->GetOutput());
	ImageWriter2->SetFileName("Transformed3Test.nrrd");

	try
	{
		ImageWriter->Update();
		ImageWriter2->Update();
	}
	catch (itk::ExceptionObject err)
	{
		std::cout << "Exception caught wile writing NRRD image!" << std::endl;
		std::cout << err << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

std::string
LoadDICOMVolume
(char* Directory)
{
	std::cout << "LoadDICOMVolume" << std::endl;

	auto VolumeIO = itk::GDCMImageIO::New();

	// Reader for Volume
	typedef itk::ImageSeriesReader< VolumeType >VolumeReaderType;
	auto VolumeReader = VolumeReaderType::New();
	VolumeReader->SetImageIO(VolumeIO);

	// Read image from directory with same aquisition date
	typedef std::vector< std::string > FileNamesContainer;
	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	auto NameGenerator = NamesGeneratorType::New();
	NameGenerator->SetUseSeriesDetails(true);
	NameGenerator->AddSeriesRestriction("0008|0021");
	NameGenerator->SetDirectory(Directory);
	auto SeriesUID = NameGenerator->GetSeriesUIDs();
	std::string seriesIdentifier = SeriesUID.begin()->c_str();
	FileNamesContainer fileNames = NameGenerator->GetFileNames(seriesIdentifier);
	VolumeReader->SetFileNames(fileNames);

	// Store input volume in Volume object
	try
	{
		VolumeReader->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while loading DICOM!" << std::endl;
		EO.Print(std::cout);
		return "error";
	}
	
	auto VolumeOrigin  = VolumeReader->GetOutput()->GetOrigin();
	auto VolumeSize    = VolumeReader->GetOutput()->GetLargestPossibleRegion().GetSize();
	auto VolumeSpacing = VolumeReader->GetOutput()->GetSpacing();
	auto VolumeDirection = VolumeReader->GetOutput()->GetDirection();

	EulerTransformType::OutputVectorType NewOrigin;
	NewOrigin.Fill(0);
	for (auto i = 0; i < 3; ++i)
	{
		NewOrigin[i] = - ( VolumeSize[i] / 2.0 * VolumeSpacing[i] );
	}
	
	// ResampleImageFilter
	auto Resampler = ResamplerFilterType::New();
	Resampler->SetOutputOrigin(NewOrigin);
	Resampler->SetOutputSpacing(VolumeSpacing);
	Resampler->SetSize(VolumeSize);
	Resampler->SetOutputDirection(VolumeDirection);
	Resampler->SetInput(VolumeReader->GetOutput());

	EulerTransformType::OutputVectorType NewTranslation;
	NewTranslation.Fill(0);
	for (auto i = 0; i < 3; ++i)
	{
		NewTranslation[i] = VolumeOrigin[i] - NewOrigin[i];
	}

	auto Transform = EulerTransformType::New();
	Transform->Translate(NewTranslation);

	Resampler->SetTransform(Transform);
	Resampler->SetDefaultPixelValue(0);

	// Image transfer function to absorption values
	/* µ_tissue = HU / 1000 * µ_water + µ_water
	*  (µ_water = 0.1707 at 100keV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
	*  A threshold image filter is applied after to suppress negative air attenuation values
	*/
	auto DivideBy1000MultiplyByWaterAttenuationFilter = MultiplyImageFilterType::New();
	DivideBy1000MultiplyByWaterAttenuationFilter->SetInput(Resampler->GetOutput());
	DivideBy1000MultiplyByWaterAttenuationFilter->SetConstant( 0.1707 / 1000);
	auto AddWaterAttentuationFilter = AddImageFilterType::New();
	AddWaterAttentuationFilter->SetInput(DivideBy1000MultiplyByWaterAttenuationFilter->GetOutput());
	AddWaterAttentuationFilter->SetConstant(0.1707);
	auto SuppressNegativeAirValuesFilter = ThresholdImageFilterType::New();
	SuppressNegativeAirValuesFilter->SetInput(AddWaterAttentuationFilter->GetOutput());
	SuppressNegativeAirValuesFilter->SetLower(0.05);
	SuppressNegativeAirValuesFilter->SetOutsideValue(0.f);
	
	// Write NRRD image to disk because of an issue between CudaImage and itk::ChangeImageInformationFilter
	std::string ImageFileName("CenteredVolume.nrrd");
	auto ImageWriter = itk::ImageFileWriter<VolumeType>::New();
	ImageWriter->SetInput(SuppressNegativeAirValuesFilter->GetOutput());
	ImageWriter->SetFileName(ImageFileName);
	try
	{
		ImageWriter->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while writing NRRD image!" << std::endl;
		EO.Print(std::cout);
		return "error";
	}

	// Write text file for image information
	std::ofstream ofstream;
	ofstream.open("CenteredVolume.txt");
	ofstream << Resampler->GetOutput()->GetOrigin() << std::endl;
	ofstream << Resampler->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
	ofstream << Resampler->GetOutput()->GetSpacing() << std::endl;
	ofstream << Resampler->GetOutput()->GetDirection() << std::endl;
	ofstream.close();

	return ImageFileName;
}

std::string
LoadNRRDVolume
(char* FileName)
{
	std::cout << "LoadNRRDVolume" << std::endl;

	auto NRRDVolumeReader = itk::ImageFileReader<VolumeType>::New();
	NRRDVolumeReader->SetFileName(FileName);

	try
	{
		NRRDVolumeReader->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while loading DICOM!" << std::endl;
		EO.Print(std::cout);
		return "error";
	}

	auto VolumeOrigin    = NRRDVolumeReader->GetOutput()->GetOrigin();
	auto VolumeSize      = NRRDVolumeReader->GetOutput()->GetLargestPossibleRegion().GetSize();
	auto VolumeSpacing   = NRRDVolumeReader->GetOutput()->GetSpacing();
	auto VolumeDirection = NRRDVolumeReader->GetOutput()->GetDirection();

	EulerTransformType::OutputVectorType NewOrigin;
	NewOrigin.Fill(0);
	for (auto i = 0; i < 3; ++i)
	{
		NewOrigin[i] = -(VolumeSize[i] / 2.0 * VolumeSpacing[i]);
	}

	// ResampleImageFilter
	auto Resampler = ResamplerFilterType::New();
	Resampler->SetOutputOrigin(NewOrigin);
	Resampler->SetOutputSpacing(VolumeSpacing);
	Resampler->SetSize(VolumeSize);
	Resampler->SetOutputDirection(VolumeDirection);
	Resampler->SetInput(NRRDVolumeReader->GetOutput());

	EulerTransformType::OutputVectorType NewTranslation;
	NewTranslation.Fill(0);
	for (auto i = 0; i < 3; ++i)
	{
		NewTranslation[i] = VolumeOrigin[i] - NewOrigin[i];
	}

	auto Transform = EulerTransformType::New();
	Transform->Translate(NewTranslation);

	Resampler->SetTransform(Transform);
	Resampler->SetDefaultPixelValue(100);

	// Image transfer function to absorption values
	/* µ_tissue = HU / 1000 * µ_water + µ_water
	*  (µ_water = 0.1707 at 100keV from https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
	*  A threshold image filter is applied after to suppress negative air attenuation values
	*/
	auto DivideBy1000MultiplyByWaterAttenuationFilter = MultiplyImageFilterType::New();
	DivideBy1000MultiplyByWaterAttenuationFilter->SetInput(Resampler->GetOutput());
	DivideBy1000MultiplyByWaterAttenuationFilter->SetConstant(0.1707 / 1000);
	auto AddWaterAttentuationFilter = AddImageFilterType::New();
	AddWaterAttentuationFilter->SetInput(DivideBy1000MultiplyByWaterAttenuationFilter->GetOutput());
	AddWaterAttentuationFilter->SetConstant(0.1707);
	auto SuppressNegativeAirValuesFilter = ThresholdImageFilterType::New();
	SuppressNegativeAirValuesFilter->SetInput(AddWaterAttentuationFilter->GetOutput());
	SuppressNegativeAirValuesFilter->SetLower(0.05);
	SuppressNegativeAirValuesFilter->SetOutsideValue(0.f);

	// Write NRRD image to disk because of an issue between CudaImage and itk::ChangeImageInformationFilter
	std::string ImageFileName("CenteredVolume.nrrd");
	auto ImageWriter = itk::ImageFileWriter<VolumeType>::New();
	ImageWriter->SetInput(SuppressNegativeAirValuesFilter->GetOutput());
	ImageWriter->SetFileName(ImageFileName);
	try
	{
		ImageWriter->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while writing NRRD image!" << std::endl;
		EO.Print(std::cout);
		return "error";
	}

	// Write text file for image information
	std::ofstream ofstream;
	ofstream.open("CenteredVolume.txt");
	ofstream << Resampler->GetOutput()->GetOrigin() << std::endl;
	ofstream << Resampler->GetOutput()->GetLargestPossibleRegion().GetSize() << std::endl;
	ofstream << Resampler->GetOutput()->GetSpacing() << std::endl;
	ofstream << Resampler->GetOutput()->GetDirection() << std::endl;
	ofstream.close();

	return ImageFileName;
}

CImageType::Pointer
LoadCUDAImage
(std::string ImageFileName)
{
	std::cout << "LoadCUDAImage" << std::endl;

	auto NRRDFileReader = itk::ImageFileReader<CImageType>::New();
	NRRDFileReader->SetFileName(ImageFileName);

	auto CudaVolume = NRRDFileReader->GetOutput();

	try
	{
		CudaVolume->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while loading CUDA image!" << std::endl;
		EO.Print(std::cout);
		return ITK_NULLPTR;
	}

	return CudaVolume;
}

GeometryType::Pointer
LoadProjectionGeometry
(char* ProjectionImageFile, DetectorGeometry &Det)
{
	std::cout << "LoadProjectionGeometry" << std::endl;

	// Allow private tags
	auto DICOMIO = itk::GDCMImageIO::New();
	DICOMIO->LoadPrivateTagsOn();

	// Reader for projection image
	auto ProjectionImageReader = ProjectionImageReaderType::New();
	ProjectionImageReader->SetImageIO(DICOMIO);
	ProjectionImageReader->SetFileName(ProjectionImageFile);

	// Get projection image to extract C-arm properties
	auto ProjectionImage = ProjectionImageReader->GetOutput();
	try
	{
		ProjectionImage->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while loading projection image!" << std::endl;
		EO.Print(std::cout);
		return ITK_NULLPTR;
	}

	auto DetectorSize = ProjectionImage->GetLargestPossibleRegion().GetSize();
	auto DetectorSpacing = ProjectionImage->GetSpacing();

	// Read DICOM header
	auto Dictionary = DICOMIO->GetMetaDataDictionary();

	/*   Extract projection geometry
	 *  The extracted projection geometry is in the patient/DICOM coordinate system (COS)
	 *  not the fixed coordinate system of the gantry system.
	 *  The values for the patient COS are tagged with "*_p",
	 *  Values for the gantry COS are tagged with *_f or *_g
	*/
	auto ProjectionGeometry = GeometryType::New();

	std::string Key_SDD("0018|1110");
	DictionaryType::ConstIterator Iter_SDD = Dictionary.Find(Key_SDD);

	std::string Key_SID("0018|1111");
	DictionaryType::ConstIterator Iter_SID = Dictionary.Find(Key_SID);

	std::string Key_PPA("0018|1510");
	DictionaryType::ConstIterator Iter_PPA = Dictionary.Find(Key_PPA);

	std::string Key_PSA("0018|1511");
	DictionaryType::ConstIterator Iter_PSA = Dictionary.Find(Key_PSA);

	std::string Key_GIO("0021|1057");
	DictionaryType::ConstIterator Iter_GIO = Dictionary.Find(Key_GIO);

	MetaDataStringType::ConstPointer SSD_ptr = dynamic_cast<const MetaDataStringType *>(Iter_SDD->second.GetPointer());
	MetaDataStringType::ConstPointer SID_ptr = dynamic_cast<const MetaDataStringType *>(Iter_SID->second.GetPointer());
	MetaDataStringType::ConstPointer PPA_ptr = dynamic_cast<const MetaDataStringType *>(Iter_PPA->second.GetPointer());
	MetaDataStringType::ConstPointer PSA_ptr = dynamic_cast<const MetaDataStringType *>(Iter_PSA->second.GetPointer());
	MetaDataStringType::ConstPointer GIO_ptr = dynamic_cast<const MetaDataStringType *>(Iter_GIO->second.GetPointer());
		
	std::cout << "SDD (" << Iter_SDD->first << ") " << " is: \t" << SSD_ptr->GetMetaDataObjectValue() << std::endl;
	std::cout << "SID (" << Iter_SID->first << ") " << " is: \t" << SID_ptr->GetMetaDataObjectValue() << std::endl;
	std::cout << "PPA (" << Iter_PPA->first << ") " << " is: \t" << PPA_ptr->GetMetaDataObjectValue() << std::endl;
	std::cout << "PSA (" << Iter_PSA->first << ") " << " is: \t" << PSA_ptr->GetMetaDataObjectValue() << std::endl;
	std::cout << "GIO (" << Iter_GIO->first << ") " << " is: \t" << GIO_ptr->GetMetaDataObjectValue() << std::endl;

	// Convert string to float and assuming default value for InPlaneAngle rotation
	auto SDD_p = std::stof(SSD_ptr->GetMetaDataObjectValue());
	auto SID_p = std::stof(SID_ptr->GetMetaDataObjectValue());
	auto PPA_p = std::stof(PPA_ptr->GetMetaDataObjectValue());
	auto PSA_p = std::stof(PSA_ptr->GetMetaDataObjectValue());
	
	std::regex delim("\\\\");
	auto GIO_tmp = GIO_ptr->GetMetaDataObjectValue();
	std::sregex_token_iterator it(GIO_tmp.begin(), GIO_tmp.end(), delim, -1);
	CImageType::PointType GIO_p;
	// The isocenter of the gantry is in 0.1mm steps
	GIO_p[2] = std::stof(it->str()) / 10; ++it;
	GIO_p[0] = std::stof(it->str()) / 10; ++it;
	GIO_p[1] = std::stof(it->str()) / 10;

	/* Transform the values from the patient COS to the gantry COS
	 * First define all gantry properties in Zero positioning in patient COS
	 * than transform to the gantry COS by rotating the patient COS -90°
	 * about the x axis.
	*/

	GeometryType::PointType Source_p;
	Source_p[0] = 0;
	Source_p[1] = SID_p;
	Source_p[2] = 0;
	GeometryType::PointType Detector_p;
	Detector_p[0] = - ( DetectorSize[0] / 2.0 * DetectorSpacing[0] );
	Detector_p[1] = SID_p - SDD_p;
	Detector_p[2] = - ( DetectorSize[0] / 2.0 * DetectorSpacing[0] );
	GeometryType::PointType Detector_u_p;
	Detector_u_p[0] = 1;
	Detector_u_p[1] = 0;
	Detector_u_p[2] = 0;
	GeometryType::PointType Detector_v_p;
	Detector_v_p[0] = 0;
	Detector_v_p[1] = 0;
	Detector_v_p[2] = 1;

	// for debugging purposes
	std::cout << Source_p << std::endl;
	std::cout << Detector_p << std::endl;
	std::cout << Detector_u_p << std::endl;
	std::cout << Detector_v_p << std::endl;
	
	// volume transformation
	auto M_p2f = EulerTransformType::New();
	M_p2f->SetRotation(DegreesToRadians(PSA_p), DegreesToRadians(0), DegreesToRadians(PPA_p));
	std::cout << "M_p2f: \n" << M_p2f->GetMatrix() << std::endl;

	auto Source_f     = M_p2f->GetMatrix() * Source_p;
	auto Detector_f   = M_p2f->GetMatrix() * Detector_p;
	auto Detector_u_f = M_p2f->GetMatrix() * Detector_u_p;
	auto Detector_v_f = M_p2f->GetMatrix() * Detector_v_p;

	// write positioning to file
	std::ofstream output;
	output.open("Gantry.txt");
	std::cout << Source_f << std::endl;
	std::cout << Detector_f << std::endl;
	std::cout << Detector_u_f << std::endl;
	std::cout << Detector_v_f << std::endl;
	output.close();

	GeometryType::VectorType u;
	u[0] = Detector_u_f[0];
	u[1] = Detector_u_f[1];
	u[2] = Detector_u_f[2];
	GeometryType::VectorType v;
	v[0] = Detector_v_f[0];
	v[1] = Detector_v_f[1];
	v[2] = Detector_v_f[2];

	Det.m_det = Detector_f;
	Det.m_src = Source_f;
	Det.m_u = u;
	Det.m_v = v;

	auto okay = ProjectionGeometry->AddProjection(Source_f, Detector_f, u, v);
	std::cout << okay << std::endl;

	std::cout << ProjectionGeometry->GetMatrices().at(0) << std::endl;
	std::cout << ProjectionGeometry->GetGantryAngles().at(0) << std::endl;
	std::cout << ProjectionGeometry->GetSourceAngles().at(0) << std::endl;
	std::cout << ProjectionGeometry->GetInPlaneAngles().at(0) << std::endl;
	std::cout << ProjectionGeometry->GetProjectionCoordinatesToDetectorSystemMatrix(0) << std::endl;
	
	return ProjectionGeometry;
}

CImageType::Pointer
SetupCudaProjectionImage
(char* ProjectionImageFile)
{
	std::cout << "SetupCudaProjectionImage" << std::endl;

	// Reader for projection image
	auto DICOMIO = itk::GDCMImageIO::New();
	
	auto ProjectionImageReader = ProjectionImageReaderType::New();
	ProjectionImageReader->SetImageIO(DICOMIO);
	ProjectionImageReader->SetFileName(ProjectionImageFile);

	// Get projection image to extract C-arm properties
	auto ProjectionImage = ProjectionImageReader->GetOutput();
	try
	{
		ProjectionImage->Update();
	}
	catch (itk::ExceptionObject EO)
	{
		std::cout << "Exception caught while loading projection image!" << std::endl;
		EO.Print(std::cout);
		return ITK_NULLPTR;
	}

	// Setup Cuda projection image aka detector properties (except angles)
	auto CudaProjectionImage = CImageType::New();
	CImageType::IndexType ProjectionImageIndex;
	ProjectionImageIndex[0] = 0;
	ProjectionImageIndex[1] = 0;
	ProjectionImageIndex[2] = 0;

	CImageType::SizeType ProjectionImageSize;
	ProjectionImageSize[0] = ProjectionImage->GetLargestPossibleRegion().GetSize()[0];
	ProjectionImageSize[1] = ProjectionImage->GetLargestPossibleRegion().GetSize()[1];
	ProjectionImageSize[2] = 1; // only one image is taken

	CImageType::RegionType ProjectionImageRegion(ProjectionImageIndex, ProjectionImageSize);
	CudaProjectionImage->SetRegions(ProjectionImageRegion);
	CudaProjectionImage->Allocate();

	CImageType::SpacingType ProjectionImageSpacing;
	ProjectionImageSpacing[0] = ProjectionImage->GetSpacing()[0];
	ProjectionImageSpacing[1] = ProjectionImage->GetSpacing()[1];
	ProjectionImageSpacing[2] = 1;
	CudaProjectionImage->SetSpacing(ProjectionImageSpacing);

	CImageType::PointType ProjectionImageOrigin;
	ProjectionImageOrigin[0] = 0;
	ProjectionImageOrigin[1] = 0;
	ProjectionImageOrigin[2] = 0;
	CudaProjectionImage->SetOrigin(ProjectionImageOrigin);

	return CudaProjectionImage;
}
