cmake_minimum_required(VERSION 3.10.2)
project(MetricDerivativeTest)

set(CMAKE_CXX_STANDARD 17)

find_package(ITK REQUIRED COMPONENTS ITKCommon ITKRegistrationCommon ITKIONRRD)
include(${ITK_USE_FILE})

set(SRCS
	${CMAKE_PROJECT_NAME}.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})