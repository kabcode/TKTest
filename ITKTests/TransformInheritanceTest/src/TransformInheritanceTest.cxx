#include "itkCudaImage.h"
#include "itkEuler3DTransform.h"
#include "itkAffineTransform.h"
#include "itkBSplineTransform.h"
#include "itkTransform.h"

static constexpr unsigned int Dim3D = 3;
using ImageType = itk::CudaImage<float, Dim3D>;
using TransformType = itk::Transform<double, Dim3D, Dim3D>;
using TransformBasePointer = TransformType::Pointer;
using TransformParametersType = TransformType::ParametersType;

class ITK_EXPORT TransformContainer
{
public:

	TransformContainer::TransformContainer()
	{
		m_Transform = nullptr;
	}

	void SetTransform(TransformBasePointer TransformPointer)
	{
		this->m_Transform = TransformPointer;
	}

	TransformBasePointer GetTransform() const
	{
		return this->m_Transform;
	}

	void AddTransformToVector(TransformBasePointer transform)
	{
		this->m_TransformVector.push_back(transform);
	}

	std::vector<TransformBasePointer> ReturnTransformVector() const
	{
		return this->m_TransformVector;
	}

	TransformBasePointer ReturnTransformVectorElement(unsigned int index) const
	{
		return this->m_TransformVector[index];
	}

protected:
	TransformBasePointer m_Transform;
	std::vector<TransformBasePointer> m_TransformVector;
};

int main(int argc, char* argv[])
{
	auto TC = TransformContainer();
	
	// Rigide 3D Transformation
	using EulerTransFormType = itk::Euler3DTransform<double>;
	auto EulerT = EulerTransFormType::New();

	TC.SetTransform(EulerT->Clone().GetPointer());
	auto RET = TC.GetTransform();
	RET->Print(std::cout);

	TC.AddTransformToVector(EulerT.GetPointer());

	// Affine 3D Transformation
	using AffineTransformType = itk::AffineTransform<double, Dim3D>;
	auto AffineT = AffineTransformType::New();

	TC.SetTransform(AffineT.GetPointer());
	auto RAT = TC.GetTransform();
	RAT->Print(std::cout);

	TC.AddTransformToVector(AffineT.GetPointer());

	// Deformierbare 3D Transformation
	using BSplineTransformType = itk::BSplineTransform<double>;
	auto BSplineT = BSplineTransformType::New();

	TC.SetTransform(BSplineT.GetPointer());
	auto BST = TC.GetTransform();
	BST->Print(std::cout);

	TC.AddTransformToVector(BSplineT.GetPointer());

	// Print Vector Transformations
	auto TV = TC.ReturnTransformVector();
	for (auto i = 0; i < TV.size(); ++i)
	{
		TV[i]->Print(std::cout);
	}

	return EXIT_SUCCESS;

}
