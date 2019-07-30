#include "itkImage.h"

#define itkInternalCloneHeader(x)												\
 virtual typename itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE		\
 {																				\
	typename itk::LightObject::Pointer loPtr = Superclass::InternalClone();			\
	std::cout << "\nCloning...\n" << std::endl;										\
	typename Self::Pointer rval = dynamic_cast<Self *>(loPtr.GetPointer());		\
	if (rval.IsNull())															\
	{																			\
		itkExceptionMacro(<< "downcast to type " << this->GetNameOfClass() << " failed.");\
	}	

#define itkSetGetMacro(x) \
	rval->Set##x(this->Get##x());

#define itkInternalCloneFooter(x) \
	return loPtr;				  \
  }

template<typename TPixelType, unsigned int VImageDimension>
class ClonedImage: public itk::Image<TPixelType, VImageDimension>
{
public:
	ITK_DISALLOW_COPY_AND_ASSIGN(ClonedImage);

	using Self = ClonedImage;
	using Superclass = itk::Image<TPixelType, VImageDimension>;
	using Pointer = itk::SmartPointer<Self>;
	using ConstPointer = itk::SmartPointer<const Self>;

	itkTypeMacro(ClonedImage, Image);
	itkNewMacro(ClonedImage);

	ClonedImage();
	~ClonedImage() override { }

protected:
	itk::LightObject::Pointer InternalClone() const ITK_OVERRIDE;
};

template <typename TPixelType, unsigned VImageDimension>
ClonedImage<TPixelType, VImageDimension>::ClonedImage()
{
}

template <typename TPixelType, unsigned VImageDimension>
itk::LightObject::Pointer
ClonedImage<TPixelType, VImageDimension>
::InternalClone() const
{
	// Default implementation just copies the parameters from
	// this to new transform.
	itk::LightObject::Pointer loPtr = Superclass::InternalClone();

	std::cout << "\n Cloning ... \n" << std::endl;

	Pointer rval = dynamic_cast<Self *>(loPtr.GetPointer());
	if (rval.IsNull())
	{
		itkExceptionMacro(<< "downcast to type " << this->GetNameOfClass() << " failed.");
	}
	itkSetGetMacro(Origin);
	itkSetGetMacro(Spacing);
	return loPtr;
}


int main()
{
	using ClonedImageType = ClonedImage<float, 3>;
	auto CloningImage = ClonedImageType::New();
	ClonedImageType::IndexType idx;
	idx.Fill(0);
	ClonedImageType::SizeType sz;
	sz.Fill(10);
	ClonedImageType::RegionType region(idx, sz);
	CloningImage->SetRegions(region);
	CloningImage->Allocate();
	ClonedImageType::PointType origin;
	origin.Fill(5);
	CloningImage->SetOrigin(origin);
	ClonedImageType::SpacingType sp;
	sp.Fill(12);
	CloningImage->SetSpacing(sp);
	CloningImage.Print(std::cout);

	auto ClonedImage = CloningImage->Clone();

	ClonedImage->Print(std::cout);

	return EXIT_SUCCESS;
}
