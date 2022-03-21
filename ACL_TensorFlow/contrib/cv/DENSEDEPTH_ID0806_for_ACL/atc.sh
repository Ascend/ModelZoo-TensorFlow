#!/bin/bash
atc --model=./pd/test.pb --input_shape="input_1:1,480,640,3" --framework=3 --output=./om/test --soc_version=Ascend910A --input_format=NHWC \
