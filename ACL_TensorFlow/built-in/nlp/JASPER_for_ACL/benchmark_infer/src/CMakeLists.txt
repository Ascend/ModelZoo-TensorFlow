# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(benchmark_inference)

# Compile options
add_compile_options(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../outputs")
set(CMAKE_INSTALL_PREFIX "../../../")
set(CMAKE_OUTPUT_DIR "../output")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -g -Wall")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

ADD_DEFINITIONS("-DENABLE_DVPP_INTERFACE")
ADD_DEFINITIONS("-DVERSION_HOST")
ADD_DEFINITIONS("-DVERSION_C75_NOT_C73")

# Header path
include_directories(
$ENV{ASCEND_HOME}/acllib/include
/usr/local/include
../inc
../inc/common
../inc/infer/base
../inc/infer/extend
../inc/vdec/base
../inc/vdec/extend
../inc/vpc/base
../inc/vpc/extend
../inc/dataset/base
../inc/dataset/extend
../inc/venc/base
../inc/venc/extend
)

# add host lib path
link_directories($ENV{ASCEND_HOME}/acllib/lib64/stub)
link_directories(/usr/local/lib)

file(GLOB_RECURSE src_files
	"./common_api/*"
	"./infer_api/*"
	"./testcase/*"
	"./vdec_api/*"
	"./vpc_api/*"
	"./dataset_api/*"
	"./venc_api/*"
)

add_executable(benchmark   
        main.cpp
        ${src_files}
)      

target_link_libraries(benchmark acl_dvpp ascendcl pthread avformat avcodec avutil swscale swresample opencv_highgui opencv_core)

install(TARGETS benchmark DESTINATION ${CMAKE_OUTPUT_DIR})

