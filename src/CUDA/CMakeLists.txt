cmake_minimum_required(VERSION 3.18.4)

set(CMAKE_CXX_STANDARD 11)

file(GLOB CUDAfiles *.cu)

add_library(poppygpu SHARED ${CUDAfiles})
target_link_libraries(poppygpu PRIVATE poppyrefl)
