cmake_minimum_required(VERSION 3.10.2)
project(CudaITK)

FIND_PACKAGE(ITK REQUIRED COMPONENTS ITKIOImageBase ITKIOTransformBase RTK ITKImageGrid ITKTransform ITKImageIO ITKTransformIO)
INCLUDE (${ITK_USE_FILE})

set( SRCS
 CudaITK.cxx
 itkCudaTransform.hcu
 itkCudaTransform.cu
)

cuda_add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})