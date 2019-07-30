cmake_minimum_required(VERSION 3.10.2)
project(ExpandImageFilterTest)
message(STATUS "Configure project: ${CMAKE_PROJECT_NAME}")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set( SRCS
	ExpandImageFilterTest.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})