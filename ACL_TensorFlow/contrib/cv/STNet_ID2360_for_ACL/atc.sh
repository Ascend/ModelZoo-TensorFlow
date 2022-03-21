#!/bin/bash
atc --model=stnet.pb --framework=3 --output=stnet_om --soc_version=Ascend310 --input_shape="Placeholder:1,1600"