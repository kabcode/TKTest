#include <iostream>

class BaseImage
{
public:
    static constexpr bool GPUCapable = false; //C++11
};

class DerivedImage : public BaseImage
{
public:
    static constexpr bool GPUCapable = true; //C++11
};

template<typename TInputImage>
class ImageFilter
{
public:
    void Run()
    {
        // C++17 version, function selection at compile time
        /*
        if constexpr (TInputImage::GPUCapable) //C++17; 
        {
            RunGPU();
        }
        else
        {
            RunCPU();
        }
        */

        // for C++11 compatibility, compiler should drop the dead code
        if (TInputImage::GPUCapable)
        {
            RunGPU();
        }
        else
        {
            RunCPU();
        }
    }

private:
    void static RunGPU() { std::cout << "Run on GPU!" << std::endl; };
    void static RunCPU() { std::cout << "Run on CPU!" << std::endl; };
};

int main()
{
  
    auto filter1 = new ImageFilter<BaseImage>();
    filter1->Run();

    auto filter2 = new ImageFilter<DerivedImage>();
    filter2->Run();

    delete filter1;
    delete filter2;

    return 0;
}