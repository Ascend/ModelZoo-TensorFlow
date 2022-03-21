#!/bin/bash
atc --model=model.pb --input_shape="input_img:1,736,736,3" --framework=3 --output=./om/modelom --soc_version=Ascend310 --input_format=NHWC \

