include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/TestLibraryConfig.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/TestLibraryConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/cmake/TestLibExport.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
)