cmake_minimum_required(VERSION 3.10.2)
project(TransformInheritanceTest)

find_package(ITK REQUIRED COMPONENTS RTK)
include(${ITK_USE_FILE})

set(SRCS
	TransformInheritanceTest.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})