cmake_minimum_required(VERSION 3.16.3)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

project(PyPO-pkg 
    VERSION 1.0.0 
    DESCRIPTION "Python Physical Optics"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_library(pyporefl SHARED src/common/InterfaceReflector.cpp)
target_include_directories(pyporefl PUBLIC src/common src/include)

add_library(pypobeam SHARED src/common/InterfaceBeam.cpp)
target_link_libraries(pypobeam PRIVATE pyporefl)

add_library(pypotransf SHARED src/common/Transform.cpp)
target_include_directories(pypotransf PRIVATE src/include)

add_library(pypocpu SHARED src/CPU/InterfaceCPU.cpp)
target_include_directories(pypocpu PRIVATE src/CPU)

if(NOT WIN32)
    target_compile_options(pypocpu PRIVATE -pthread)
endif()

target_link_libraries(pypocpu PRIVATE pyporefl)

include(CheckLanguage)

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    file(GLOB CUDAfiles src/CUDA/*.cu)

    add_library(pypogpu SHARED ${CUDAfiles})
    target_include_directories(pypogpu PRIVATE src/CUDA)
    target_link_libraries(pypogpu PRIVATE pyporefl)
    set_target_properties(pypogpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET pypogpu PROPERTY CUDA_ARCHITECTURES OFF)
else()
    message(STATUS "WARNING: no CUDA compiler detected. Not building CUDA libraries")
endif()



