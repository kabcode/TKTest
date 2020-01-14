#include <iostream>
#include "itkCudaImage.h"
#include "itkCudaDataManager.h"

#include <chrono>

using namespace std::chrono;

void Foo(const unsigned int i)
{
    std::string* moreInformation = nullptr;
    moreInformation = new std::string();
    auto res = (*moreInformation).empty();

    if (i % 1000 == 0)
    {
        moreInformation = new std::string();
        *moreInformation = "Hello!";
    }


    delete moreInformation;
}

int main(int argc, char* argv[]  )
{
    Foo(1);


    auto CudaImage = itk::CudaImage<float, 3>::New();
    auto DataManager = CudaImage->GetDataManager();

    std::vector<double> durations{ 1000 };

    for (auto& duration : durations)
    {
        auto Start = high_resolution_clock::now();
        for (unsigned int i = 0; i < 1000000; ++i)
        {
            DataManager->UpdateCPUBuffer();
        }
        auto Stop = high_resolution_clock::now();

        duration = duration_cast<milliseconds>(Stop - Start).count();
    }
    
    double sum_of_elems = 0;
    for (auto& n : durations)
        sum_of_elems += n;

    std::cout << sum_of_elems / durations.size() << "ms." << std::endl;
    
	return EXIT_SUCCESS;
}
