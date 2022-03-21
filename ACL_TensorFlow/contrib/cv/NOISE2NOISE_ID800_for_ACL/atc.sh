#!/bin/bash
atc --model=./pb/test.pb --input_shape="input:1,3,512,768" --framework=3 --output=./om/test --soc_version=Ascend910A --input_format=NCHW
atc --model=./pb/test_mri.pb --input_shape="input:1,255,255" --framework=3 --output=./om/test_mri --soc_version=Ascend910A