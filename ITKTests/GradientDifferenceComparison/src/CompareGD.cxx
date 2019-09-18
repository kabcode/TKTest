#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEuler3DTransform.h"

#include "itkGradientDifferenceImageToImageMetric.h"
#include "itkAdvGradientDifferenceImageToImageMetric.h"


using ImageType = itk::Image<double, 3>;
using VectorImageType = itk::Image<itk::CovariantVector<double, 3>, 3>;


int main(int argc, char* argv[])
{
    auto FileReader = itk::ImageFileReader<ImageType>::New();
    FileReader->SetFileName(argv[1]);

    auto image1 = FileReader->GetOutput();
    image1->Update();

    auto image2 = FileReader->GetOutput();
    image2->Update();

    // add components
    auto InterpolatorFunction = itk::LinearInterpolateImageFunction<ImageType>::New();
    auto Transform = itk::Euler3DTransform<>::New();
    itk::Euler3DTransform<>::ParametersType parameters(Transform->GetNumberOfParameters());
    parameters.Fill(0);

    auto GDMetric = itk::GradientDifferenceImageToImageMetric<ImageType, ImageType>::New();
    GDMetric->SetInterpolator(InterpolatorFunction);
    GDMetric->SetTransform(Transform);

    GDMetric->SetFixedImage(image1);
    GDMetric->SetFixedImageRegion(image1->GetLargestPossibleRegion());
    GDMetric->SetMovingImage(image2);

    GDMetric->Initialize();
    auto GDValue = GDMetric->GetValue(parameters);

    auto AGDMetric = itk::AdvGradientDifferenceImageToImageMetric<ImageType, ImageType>::New();
    AGDMetric->SetInterpolator(InterpolatorFunction);
    AGDMetric->SetTransform(Transform);

    AGDMetric->SetFixedImage(image1);
    AGDMetric->SetFixedImageRegion(image1->GetLargestPossibleRegion());
    AGDMetric->SetMovingImage(image2);

    AGDMetric->Initialize();
    auto AGDValue = AGDMetric->GetValue(parameters);

    //AGDMetric->GetModifiableGradientImage()->Update();
    GDMetric->GetModifiableGradientImage()->Update();

    auto SubtractionImageFilter = itk::SubtractImageFilter<VectorImageType, VectorImageType>::New();
    SubtractionImageFilter->SetInput1(AGDMetric->GetModifiableGradientImage());
    SubtractionImageFilter->SetInput2(GDMetric->GetModifiableGradientImage());

    auto ImageWriter = itk::ImageFileWriter<VectorImageType>::New();
    ImageWriter->SetInput(SubtractionImageFilter->GetOutput());
    ImageWriter->SetFileName("GDiffImage.nrrd");
    try
    {
        ImageWriter->Update();
    }
    catch (itk::ExceptionObject &EO)
    {
        EO.Print(std::cout);
    }
    

    std::cout << "GD:  " << GDValue << std::endl;
    std::cout << "AGD: " << AGDValue << std::endl;
}