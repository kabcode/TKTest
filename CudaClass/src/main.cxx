#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "CudaHost.hcu"

auto main(int argc, char* argv[]) -> int
{
    auto CudaHostClass = new CudaHost();
    CudaHostClass->Run();
    delete CudaHostClass;

    return EXIT_SUCCESS;
}
