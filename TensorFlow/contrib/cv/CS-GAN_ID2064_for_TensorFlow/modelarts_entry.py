import os
import argparse
import sys

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/cache/dataset")
parser.add_argument("--train_url", type=str, default="/cache/output")
config = parser.parse_args()

print("[CANN-ZhongZhi] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]

print("[CANN-ZhongZhi] work_dir path is [%s]" % (os.getcwd()))
work_dir = os.getcwd()

print("[CANN-ZhongZhi] start run train shell")
# 执行训练脚本
shell_cmd = ("bash %s/npu_train.sh %s %s %s %s " % (code_dir, code_dir, work_dir, config.data_url, config.train_url))
os.system(shell_cmd)
print("[CANN-ZhongZhi] finish run train shell")




