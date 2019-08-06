#include <cstdlib>
#include <algorithm>

#include "cudaclass.hcu"

int main(int argc, char* argv[])
{
    auto cudaclass = new CudaClass<2>;

    unsigned int size[] = { 5,10 };

    cudaclass->SetSize(size);


    delete cudaclass;

    return EXIT_SUCCESS;
}
