#!/bin/bash
atc --model=./data/pb/symnet_new.pb --framework=3 --output=./data/om/symnet --soc_version=Ascend310 --input_shape="Placeholder_2:1,512;test_attr_id:116;test_obj_id:116;Placeholder_6:1,12" --out_nodes="Mul_18:0;Softmax_3:0;Placeholder_6:0"

