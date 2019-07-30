cmake_minimum_required(VERSION 3.10)

project(TestUseLibrary)

set(SRCS
	UseDummy.cxx
)

add_executable(TestUseLibrary ${SRCS})
target_link_libraries(TestUseLibrary 
	TestLibrary
)