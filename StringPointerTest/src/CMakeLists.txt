cmake_minimum_required (VERSION 3.10)

project(StringPointerTest)

set(CMAKE_CXX_STANDARD 17)

set (SRCS
	${CMAKE_PROJECT_NAME}.cxx
)

add_executable(${CMAKE_PROJECT_NAME}
	${SRCS}
)