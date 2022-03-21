#!/bin/bash
atc --model=./c3ae_npu_train_v2.pb --framework=3 --output=om_C3AE --soc_version=Ascend310 --input_shape="input_2:1,64,64,3;input_3:1,64,64,3;input_4:1,64,64,3"
