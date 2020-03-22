cmake_minimum_required(VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(CudaImageTransfer)
set(CMAKE_CXX_STANDARD 17)

# Find the RTK libraries and includes
find_package(ITK REQUIRED COMPONENTS ITKCommon RTK ITKCudaCommon MultiObjectRegistrationFramework)
include(${ITK_USE_FILE})

# Executable
add_executable(${CMAKE_PROJECT_NAME} ${CMAKE_PROJECT_NAME}.cxx )
target_link_libraries(${CMAKE_PROJECT_NAME} ${RTK_LIBRARIES} ${ITK_LIBRARIES})
