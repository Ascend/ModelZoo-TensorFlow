export PYTHONPATH=$PYTHONPATH:$PWD
cd gcmc; python3 train.py -d douban --accum stack -do 0.7 -nlef -nb 2 -e 100 --features --feat_hidden 64