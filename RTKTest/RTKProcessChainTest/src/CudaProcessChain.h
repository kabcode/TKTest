#ifndef CUDAPROCESSCHAIN_H
#define CUDAPROCESSCHAIN_H

template<unsigned int TSize = 4>
class CudaChainProcess
{
public:
    
    void Run();

    auto setIn(int i) ->void;
    auto getOut() const { return out; }

private:
    int in = 0;
    int out = 0;
};

#  include "CudaProcessChain.hxx"

#endif // CUDAPROCESSCHAIN_H
