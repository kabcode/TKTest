cmake_minimum_required(VERSION 3.10.2)
project(CloneMacroTest)

FIND_PACKAGE(ITK REQUIRED)
INCLUDE (${ITK_USE_FILE})

set( SRCS
 CloneMacroTest.cxx
)

add_executable(${CMAKE_PROJECT_NAME} ${SRCS})
TARGET_LINK_LIBRARIES(${CMAKE_PROJECT_NAME} ${ITK_LIBRARIES})