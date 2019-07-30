cmake_minimum_required (VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(TestProvidedGeometry)

# Find the RTK libraries and includes
find_package(RTK REQUIRED)
include(${RTK_USE_FILE})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Executable
add_executable(TestProvidedGeometry TestProvidedGeometry.cxx )
target_link_libraries(TestProvidedGeometry
	${RTK_LIBRARIES}
	${ITK_LIBRARIES}
	${VTK_LIBRARIES}
)
