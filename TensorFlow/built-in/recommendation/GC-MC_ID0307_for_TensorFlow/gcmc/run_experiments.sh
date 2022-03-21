#!/bin/bash

for i in {604..1000};
do
# Movielens 100K on official split with features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e $i --features --feat_hidden 10 --testing
done > ml_100k_feat_testing.txt  2>&1

