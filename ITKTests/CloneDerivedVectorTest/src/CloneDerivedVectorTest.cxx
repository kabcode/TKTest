#include "itkCudaImage.h"
#include "itkImageToImageMetric.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkEuler3DTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineTransform.h"

#include "itkImageFileWriter.h"

static constexpr unsigned int Dim3D = 3;
using ImageType = itk::CudaImage<float, Dim3D>;
using MetricBaseType = itk::ImageToImageMetric<ImageType, ImageType>;
using MetricBasePointer = MetricBaseType::Pointer;

using TransformBaseType = itk::Transform<double>;
using TransformBasePointer = TransformBaseType::Pointer;
using TransformParameters = TransformBaseType::ParametersType;

template<typename T>
void SetTransformMaster(TransformBasePointer master, T transform);

int main(int argc, char* argv[])
{

	using MSMetricType = itk::MeanSquaresImageToImageMetric<ImageType, ImageType>;
	auto  MSMetric = MSMetricType::New();

	using MIMetricType = itk::MattesMutualInformationImageToImageMetric<ImageType, ImageType>;
	auto MIMetric = MIMetricType::New();

	std::vector<MetricBasePointer> m_MetricVector;
	auto m_MetricMaster = MIMetricType::New();
	m_MetricMaster->SetNumberOfHistogramBins(500);

	m_MetricVector.push_back(MSMetric.GetPointer());
	m_MetricVector.push_back(nullptr);
	m_MetricVector.push_back(MIMetric.GetPointer());
	m_MetricVector.push_back(nullptr);
	m_MetricVector.push_back(nullptr);

	// fill the vector positions that hold a nullptr
	for (auto i = 0; i < m_MetricVector.size(); ++i)
	{
		if(m_MetricVector.at(i) == nullptr)
		{
			m_MetricVector.at(i) = m_MetricMaster->Clone();
		}
	}
	
	m_MetricMaster = nullptr;

	for (auto i = 0; i < m_MetricVector.size(); ++i)
	{
		std::cout << i << ". " << m_MetricVector.at(i)->GetNameOfClass() << std::endl;
		if (dynamic_cast<MIMetricType*>(m_MetricVector.at(i).GetPointer()) != nullptr)
		{
			auto n = dynamic_cast<MIMetricType*>(m_MetricVector.at(i).GetPointer())->GetNumberOfHistogramBins();
			std::cout << "NumberOfHistrogramBins: " << n << std::endl;
		}
	}

	/********************************************/
	auto master = TransformBaseType::New();

	using EulerTransformType = itk::Euler3DTransform<double>;
	auto EulerTransform = EulerTransformType::New();
	SetTransformMaster<EulerTransformType::Pointer>(master, EulerTransform);

	using AffineTransformType = itk::AffineTransform<double>;
	auto AffineTransform = AffineTransformType::New();

	std::vector<TransformBasePointer> m_TransformVector;
	auto m_TransformMaster = EulerTransformType::New();
	m_TransformMaster->SetComputeZYX(true);

	std::cout << EulerTransform->GetNameOfClass() << ": ";
	std::cout << EulerTransform->GetNumberOfParameters() << std::endl;
	std::cout << EulerTransform->GetTransformCategory() << " : " << EulerTransform->IsLinear() << std::endl;

	EulerTransformType::ParametersType InitialParametersEuler;
	InitialParametersEuler.SetSize(EulerTransform->GetNumberOfParameters());
	InitialParametersEuler.Fill(2);
	std::cout << InitialParametersEuler << std::endl;
	EulerTransform->SetParameters(InitialParametersEuler);
	//EulerTransform->Print(std::cout);
	EulerTransform->SetIdentity();
	InitialParametersEuler = EulerTransform->GetParameters();
	std::cout << InitialParametersEuler << std::endl;

	std::cout << AffineTransform->GetNameOfClass() << ": ";
	std::cout << AffineTransform->GetNumberOfParameters() << std::endl;
	std::cout << AffineTransform->GetTransformCategory() << " : " << AffineTransform->IsLinear() << std::endl;
	//AffineTransform->Print(std::cout);

	AffineTransformType::ParametersType InitialParameters;
	InitialParameters.SetSize(AffineTransform->GetNumberOfParameters());
	InitialParameters.Fill(0);
	std::cout << InitialParameters << std::endl;
	AffineTransform->SetParameters(InitialParameters);
	//AffineTransform->Print(std::cout);
	AffineTransform->SetIdentity();
	InitialParameters = AffineTransform->GetParameters();
	std::cout << InitialParameters << std::endl;
	
	using BSplineTransformType = itk::BSplineTransform<double,3,3>;
	auto BSplineTransform = BSplineTransformType::New();

	std::cout << BSplineTransform->GetNameOfClass() << ": ";
	std::cout << BSplineTransform->GetNumberOfParameters() << std::endl;
	//BSplineTransform->Print(std::cout);
	std::cout << BSplineTransform->GetTransformCategory() << " : " << BSplineTransform->IsLinear() << std::endl;

	BSplineTransformType::ParametersType BSInitialParameters;
	BSInitialParameters.SetSize(BSplineTransform->GetNumberOfParameters());
	BSInitialParameters.Fill(0);
	//std::cout << BSplineTransform << std::endl;

	m_TransformVector.push_back(AffineTransform.GetPointer());
	m_TransformVector.push_back(nullptr);
	m_TransformVector.push_back(BSplineTransform.GetPointer());
	m_TransformVector.push_back(nullptr);
	m_TransformVector.push_back(EulerTransform.GetPointer());

	std::vector<TransformParameters> ParametersVector;
	ParametersVector.push_back(AffineTransform->GetParameters());
	ParametersVector.push_back(TransformParameters());
	ParametersVector.push_back(BSplineTransform->GetParameters());
	ParametersVector.push_back(TransformParameters());
	ParametersVector.push_back(EulerTransform->GetParameters());

	for (auto i = 0; i < ParametersVector.size(); ++i)
	{
		if (ParametersVector[i].empty())
		{
			std::cout << i << ": Nothing to see here. Size:" << ParametersVector[i].Size() << ".";
			ParametersVector[i].SetSize(i);
			std::cout << " Resized to " << ParametersVector[i].GetSize() << std::endl;
		}
		else
		{
			std::cout << i << ": " << ParametersVector[i] << std::endl;
		}

	}

	for (auto i = 0; i < m_TransformVector.size(); ++i)
	{
		if (m_TransformVector.at(i) == nullptr)
		{
			m_TransformVector.at(i) = m_TransformMaster->Clone();
			ParametersVector.at(i) = m_TransformVector.at(i)->GetParameters();
		}
	}

	for (auto i = 0; i < ParametersVector.size(); ++i)
	{
		if (ParametersVector[i].empty())
		{
			std::cout << i << ": Nothing to see here. Size:" << ParametersVector[i].Size() << std::endl;
		}
		else
		{
			std::cout << i << ": " << ParametersVector[i] << std::endl;
		}

	}

	auto CoefficientImages = BSplineTransform->GetCoefficientImages();
	auto Writer = itk::ImageFileWriter<itk::Image<double, 3>>::New();
	std::string Filename("co");
	std::string Extension(".nrrd");
	for (auto i = 0; i < CoefficientImages.Size(); ++i)
	{
		Writer->SetInput(CoefficientImages.GetElement(i));
		auto Name = Filename + std::to_string(i) + Extension;
		Writer->SetFileName(Name);
		try
		{
			Writer->Update();
		}
		catch(itk::ExceptionObject &EO)
		{
			EO.Print(std::cout);
		}
		
	}
	


	return EXIT_SUCCESS;

}

template<typename T>
void
SetTransformMaster(TransformBasePointer master, T transform)
{
	
}
