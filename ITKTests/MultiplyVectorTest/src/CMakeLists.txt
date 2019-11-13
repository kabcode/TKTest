cmake_minimum_required(VERSION 3.10.2)
project(MultiplyVectorTest)

find_package(ITK REQUIRED COMPONENTS ITKCommon ITKIOMeta
	ITKImageGradient ITKIONRRD
)
include(${ITK_USE_FILE})

set(SRCS
	MultiplyVectorTest.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})