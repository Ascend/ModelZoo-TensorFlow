export install_path=/usr/local/Ascend/latest
export DDK_PATH=/usr/local/Ascend/
export NPU_HOST_LIB=${install_path}/acllib/lib64/stub

g="g++"
cd Benchmark
rm -rf build
mkdir -p build/intermediates/host
cd build/intermediates/host
cmake ../../../src -DCMAKE_CXX_COMPILER=${g} -DCMAKE_SKIP_RPATH=TRUE
make
