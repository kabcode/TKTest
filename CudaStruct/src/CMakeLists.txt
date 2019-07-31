cmake_minimum_required(VERSION 3.14)
project(cudastruct LANGUAGES CXX)

find_package(CUDA 10.1 REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_CXX_STANDARD 17)

cuda_add_executable(cudastruct	
	cudastruct.hcu main.cxx cudakernel.cu)