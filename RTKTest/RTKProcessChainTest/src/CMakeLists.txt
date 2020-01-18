cmake_minimum_required(VERSION 3.10.2)
project(RTKProcessChainTest CXX CUDA)

find_package(ITK REQUIRED COMPONENTS ITKCommon)
include(${ITK_USE_FILE})

set( SRCS
	${CMAKE_PROJECT_NAME}.cxx
	CudaProcessChain.h
	CudaProcessChain.hxx
	CudaProcessChain.hcu
	CudaProcessChain.cu
	Reduce_kernel.hcu
)

cuda_add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES} ${ITK_LIBRARIES})