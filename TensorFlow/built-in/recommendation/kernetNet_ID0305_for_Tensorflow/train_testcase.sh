# 3、赛题三：利用Tensorflow实现kernelNet推荐网络
# 论文：
# Sample-Efficient Neural Architecture Search by Learning Action Space for Monte Carlo Tree Search
# Paperwithcode：
# https://paperswithcode.com/paper/kernelized-synaptic-weight-matrices
# Github：
# https://github.com/lorenzMuller/kernelNet_MovieLens
# 数据集：
# MovieLes
# 精度基线：
# Val rmse:82.3%
# 建议参数：train step:1000*epoch BatchSize:50


pip install tensorflow-determinism

export EXPERIMENTAL_DYNAMIC_PARTITION=1

starttime=`date +'%Y-%m-%d %H:%M:%S'`
# 执行程序 
python kernelNet_ml1m.py >train.log 2>&1
endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s)
end_seconds=$(date --date="$endtime" +%s)
echo "本次运行时间： "$((end_seconds-start_seconds))"s"

cat train.log | grep 'The Result End ' -B 4