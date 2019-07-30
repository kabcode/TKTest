cmake_minimum_required (VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(TestSetupGeometry)

# Find the RTK libraries and includes
find_package(ITK COMPONENTS RTK)
include(${ITK_USE_FILE})

# Executable
add_executable(TestSetupGeometry TestSetupGeometry.cxx )
target_link_libraries(TestSetupGeometry ${RTK_LIBRARIES})
target_link_libraries(TestSetupGeometry ${ITK_LIBRARIES})
