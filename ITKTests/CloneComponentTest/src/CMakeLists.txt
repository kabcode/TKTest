cmake_minimum_required(VERSION 3.10.2)
project(CloneComponentTest)

find_package(ITK REQUIRED COMPONENTS ITKCommon RTK)
include(${ITK_USE_FILE})

set(SRCS
	CloneComponentTest.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})