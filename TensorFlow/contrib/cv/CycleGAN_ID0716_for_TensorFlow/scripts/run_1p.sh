#!/bin/bash
nohup python3 ../train.py --result=../../result \
	--dataset=../../data/tfrecords/horse2zebra \
	--chip=npu \
	--platform=apulis \
	--train_epochs=100 \
	> ../log/train.log 2>&1 &
