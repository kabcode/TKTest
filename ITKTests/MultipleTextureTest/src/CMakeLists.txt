cmake_minimum_required(VERSION 3.10.2)
project(MultipleTextureTest)

find_package(ITK REQUIRED COMPONENTS ITKCommon RTK ITKIONRRD)
include(${ITK_USE_FILE})

set(CUDA_SRCS
  VectorKernel.cu
  )
CUDA_COMPILE(CUDA_WRAPPERS ${CUDA_SRCS})

set( SRCS
	${CMAKE_PROJECT_NAME}.cxx
	${CUDA_SRCS}
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS} ${CUDA_WRAPPERS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})