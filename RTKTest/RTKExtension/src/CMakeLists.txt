cmake_minimum_required (VERSION 3.10)

# This project is designed to be built outside the RTK source tree.
project(RTKExtension)

# Find the RTK libraries and includes
find_package(ITK COMPONENTS ITKCudaCommon RTK)
include(${ITK_USE_FILE})

set(TEST_SRCS
    CudaGradientImageFilterTest.cxx
)

# Executable
add_executable(${CMAKE_PROJECT_NAME} ${TEST_SRCS} )
target_link_libraries(${CMAKE_PROJECT_NAME}
	${ITK_LIBRARIES}
    RTK
)
