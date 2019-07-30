cmake_minimum_required (VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(Test2DImageIn3DSpace)

# Find the RTK libraries and includes
find_package(RTK REQUIRED)
include(${RTK_USE_FILE})

# Executable
add_executable(Test2DImageIn3DSpace Test2DImageIn3DSpace.cxx )
target_link_libraries(Test2DImageIn3DSpace ${RTK_LIBRARIES})
target_link_libraries(Test2DImageIn3DSpace ${ITK_LIBRARIES})
