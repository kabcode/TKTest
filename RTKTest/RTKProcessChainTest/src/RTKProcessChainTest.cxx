#include <ostream>

#include "CudaProcessChain.h"

int main(int argc, char* argv[])
{

    auto Chain = new CudaChainProcess<5>();
    Chain->Run();

    delete Chain;

    return EXIT_SUCCESS;
}
