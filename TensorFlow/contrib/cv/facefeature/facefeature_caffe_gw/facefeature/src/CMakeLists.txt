# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(classification)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)
set(INC_PATH $ENV{DDK_PATH})
set(OPENCV_PATH "/home/HwHiAiUser/ascend_ddk/arm")

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

if (${CMAKE_CXX_COMPILER} MATCHES "aarch")
    set(ARCH "arm")
else ()
    set(ARCH "x86")
endif()


# Header path
include_directories(
    $ENV{HOME}/ascend_ddk/${ARCH}/include/
    ${OPENCV_PATH}/include/opencv4
    ${INC_PATH}/acllib/include/
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${LIB_PATH}
    ${OPENCV_PATH}/lib/
    ${INC_PATH}/atc/lib64
)

add_executable(main
        utils.cpp
        model_process.cpp
        face_feature.cpp
        save_feature.cpp
        normalization.cpp
        main.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl stdc++ opencv_core opencv_imgproc opencv_imgcodecs dl rt)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})