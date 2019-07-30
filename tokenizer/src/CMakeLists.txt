cmake_minimum_required (VERSION 3.10)

# This project is designed to be built outside the RTK source tree.
project(FileTokenizer)

set (SRCS
	${CMAKE_PROJECT_NAME}.cxx
)

add_executable(${CMAKE_PROJECT_NAME}
	${SRCS}
)