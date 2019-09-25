#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkEuler3DTransform.h"

#include "itkGradientDifferenceImageToImageMetric.h"
#include "itkAdvGradientDifferenceImageToImageMetric.h"

#include "itkSobelOperator.h"
#include "itkNeighborhoodOperatorImageFilter.h"


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
    //auto GDValue = GDMetric->GetValue(parameters);

    auto GradientImageFilter = itk::GradientImageFilter<ImageType, double, double>::New();
    GradientImageFilter->SetInput(image1);
    auto AdvImageFileWriter = itk::ImageFileWriter<VectorImageType>::New();
    AdvImageFileWriter->SetInput(GradientImageFilter->GetOutput());
    AdvImageFileWriter->SetFileName("vec1ADG.nrrd");
    AdvImageFileWriter->Update();

    auto FixedGradientImage = GDMetric->GetGradientImages(true);
    auto ImageFileWriter = itk::ImageFileWriter<VectorImageType>::New();
    ImageFileWriter->SetInput(FixedGradientImage);
    ImageFileWriter->SetFileName("vec1GD.nrrd");
    ImageFileWriter->Update();

    auto AGDMetric = itk::AdvGradientDifferenceImageToImageMetric<ImageType, ImageType>::New();
    AGDMetric->SetInterpolator(InterpolatorFunction);
    AGDMetric->SetTransform(Transform);

    AGDMetric->SetFixedImage(image1);
    AGDMetric->SetFixedImageRegion(image1->GetLargestPossibleRegion());
    AGDMetric->SetMovingImage(image2);

    AGDMetric->Initialize();
    auto AGDValue = AGDMetric->GetValue(parameters);

    //std::cout << "GD:  " << GDValue << std::endl;
    std::cout << "AGD: " << AGDValue << std::endl;
    //std::cout << "Diff: " << abs(AGDValue - GDValue) << std::endl;
}