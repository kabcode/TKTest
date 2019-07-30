cmake_minimum_required(VERSION 3.10)
project(TestLibrary LANGUAGES CXX)

find_package(ITK REQUIRED COMPONENTS ITKImageIO)
include(${ITK_USE_FILE})

include(GenerateExportHeader)

set(SRCS
	src/Dummy.cxx
)

set(HDRS
	include/Dummy.h
)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(TestLibrary SHARED ${SRCS} ${HDRS})
generate_export_header(TestLibrary
	EXPORT_MACRO_NAME TESTLIB_API
	EXPORT_FILE_NAME ${CMAKE_BINARY_DIR}/include/testlibrary_export.h
)


target_include_directories(TestLibrary
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
		$<INSTALL_INTERFACE:include>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(TestLibrary PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_link_libraries(TestLibrary PUBLIC ${ITK_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS TestLibrary
	EXPORT TestLibExport
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT TestLibExport
    FILE TestLibExport.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)

export(EXPORT TestLibExport
    FILE ${CMAKE_BINARY_DIR}/cmake/TestLibExport.cmake
)
