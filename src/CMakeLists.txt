cmake_minimum_required(VERSION 3.18.4)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

project(POPPy 
    VERSION 0.5.0 
    DESCRIPTION "Physical OPtics for Python"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)

add_library(poppyrefl SHARED common/InterfaceReflector.cpp)
target_include_directories(poppyrefl PUBLIC common include)

add_library(poppybeam SHARED common/InterfaceBeam.cpp)
target_link_libraries(poppybeam PRIVATE poppyrefl)

add_library(poppycpu SHARED CPU/InterfaceCPU.cpp)

if(NOT WIN32)
    target_compile_options(poppycpu PRIVATE -pthread)
endif()

target_link_libraries(poppycpu PRIVATE poppyrefl)

include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    file(GLOB CUDAfiles CUDA/*.cu)

    add_library(poppygpu SHARED ${CUDAfiles})
    target_link_libraries(poppygpu PRIVATE poppyrefl)
else()
    message(STATUS "WARNING: no CUDA compiler detected.")
endif()

#add_library(libpoppycpu SHARED  )


