cmake_minimum_required(VERSION 3.16.3)
 
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()

include(GoogleTest)
 
# Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests include/tests/test_Utils.cpp)
target_link_libraries(runTests ${GTEST_LIBRARIES} GTest::gtest_main)
set_target_properties(runTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
if(NOT WIN32)
    target_link_libraries(runTests pthread)
endif()
gtest_discover_tests(runTests)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    add_executable(runCUDATests include/tests/test_GUtils.cpp)
    set_target_properties(runCUDATests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
    
    target_link_libraries(runCUDATests ${GTEST_LIBRARIES} GTest::gtest_main)

    if(NOT WIN32)
        target_link_libraries(runCUDATests pthread)
    endif()
    gtest_discover_tests(runCUDATests)
else()
    message(STATUS "WARNING: no CUDA compiler detected. Not building CUDA tests")
endif()

