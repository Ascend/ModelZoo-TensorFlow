#!/bin/bash
### Do not need to Configure CANN Environment on Modelarts Platform, because it has been set already.
### Modelarts Platform command for train

/usr/local/Ascend/ascend-toolkit/latest/atc/bin/atc --model=/usr/model_test/frozen_model.pb --framework=3 --output=/usr/model_test/frozen_model
--soc_version=Ascend310 --out_nodes="gen_1/Sigmoid:0" --input_shape "x_ph:100,784;x_ph_1:100,20"