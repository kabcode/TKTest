cmake_minimum_required (VERSION 3.10)

# This project is designed to be built outside the RTK source tree.
project(InterleavingArray)

#set(CMAKE_CXX_STANDARD 14)

set (SRCS
	${CMAKE_PROJECT_NAME}.cxx
	obj.h
	obj.cxx
)

add_executable(${CMAKE_PROJECT_NAME}
	${SRCS}
)

find_package(OpenMP REQUIRED) # Find the package
target_link_libraries(${PROJECT_NAME} ${OpenMP_CXX_LIBRARIES}) # Link against it for C++