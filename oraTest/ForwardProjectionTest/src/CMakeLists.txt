cmake_minimum_required (VERSION 3.0)

# This project is designed to be built outside the RTK source tree.
project(TestForwardProjection)

# add ora libraries
find_package(ORA REQUIRED)


# Executable
add_executable(TestForwardProjection ForwardProjectionTest.cxx )
target_link_libraries(TestForwardProjection ${RTK_LIBRARIES})
target_link_libraries(TestForwardProjection ${ITK_LIBRARIES})
