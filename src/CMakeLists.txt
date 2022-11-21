cmake_minimum_required(VERSION 3.18.4)

project(POPPy VERSION 0.5.0 DESCRIPTION "Physical OPtics for Python")

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(common)
add_subdirectory(CPU)
add_subdirectory(CUDA)

add_library(libpoppycpu SHARED  )


