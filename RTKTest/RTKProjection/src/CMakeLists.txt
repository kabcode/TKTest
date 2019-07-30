cmake_minimum_required(VERSION 3.10.2)
project(RTKProjection)

find_package(ITK REQUIRED COMPONENTS ITKCommon ITKIOMeta RTK ITKIOImageBase ITKIONRRD)
include(${ITK_USE_FILE})


set( SRCS RTKProjection.cxx)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})