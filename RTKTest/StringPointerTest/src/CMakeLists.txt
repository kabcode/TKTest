cmake_minimum_required (VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(StringPointerTest)

# Find the RTK libraries and includes
find_package(ITK COMPONENTS RTK)
include(${ITK_USE_FILE})

# Executable
add_executable(StringPointerTest StringPointerTest.cxx )
target_link_libraries(StringPointerTest
	${ITK_LIBRARIES}
)

