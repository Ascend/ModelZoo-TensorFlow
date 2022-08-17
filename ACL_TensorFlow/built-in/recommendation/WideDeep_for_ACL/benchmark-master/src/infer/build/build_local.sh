#!/bin/bash
export ACL_ROOT_PATH=/usr/local/Ascend/

if [ ! -d cmake_tmp ];then	
    mkdir cmake_tmp
fi
cd cmake_tmp
cmake ../../src
make
cd ../../
if [ ! -d bin ];then	
    mkdir bin
fi
cp build/cmake_tmp/benchmark bin/benchmark
cp bin/benchmark ../../../benchmark
rm -rf build/cmake_tmp