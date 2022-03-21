#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train

atc --model=/usr/model_test/dmsp_frozen_model.pb --framework=3 --output=/usr/model_test/dmsp_frozen_model
--soc_version=Ascend310 --out_nodes="strided_slice_1:0" --input_shape "input_image:1,180,180,3"