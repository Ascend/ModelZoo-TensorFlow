#!/bin/bash

CUR_DIR=$(dirname $(readlink -f $0))

mkdir -p $CUR_DIR/src/infer/build
cd $CUR_DIR/src/infer/build
./build_local.sh
cp $CUR_DIR/src/infer/bin/benchmark $CUR_DIR/../