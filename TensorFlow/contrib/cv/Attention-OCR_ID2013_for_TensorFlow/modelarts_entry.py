import os
import argparse
import sys
import datetime
import moxing as mox

def obs_data2modelarts(config):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    # if not os.path.exists(config.modelarts_ckpt_dir):
    #     os.makedirs(config.modelarts_ckpt_dir)
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(config.ckpt_path, config.modelarts_ckpt_dir))
    # mox.file.copy_parallel(src_url=config.ckpt_path, dst_url=config.modelarts_ckpt_dir)
    mox.file.copy(src_url=config.ckpt_path, dst_url=config.modelarts_ckpt_dir)
    # mox.file.copy('obs://bucket_name/obs_file.txt', '/tmp/obs_file.txt')
    end = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end-start).seconds))
    # files = os.listdir(config.modelarts_ckpt_dir)
    # print("===>>>Files", files)

# 解析输入参数data_url
parser = argparse.ArgumentParser()
parser.add_argument("--data_url", type=str, default="/cache/dataset")
parser.add_argument("--train_url", type=str, default="/cache/output")
parser.add_argument("--ckpt_path", type=str, default="obs://cann-2021-10-21/attention_ocr/attention_ocr/python/inception_v3.ckpt")
parser.add_argument("--modelarts_ckpt_dir", type=str, default="/cache/check_point.ckpt")

config = parser.parse_args()
obs_data2modelarts(config)
print(os.path.exists('/cache/check_point/inception_v3.ckpt'))

print("My_data_url:{}, My_train_url:{}, My_ckpt_path:{}".format(config.data_url, config.train_url, config.modelarts_ckpt_dir))
print("[CANN-ZhongZhi] code_dir path is [%s]" % (sys.path[0]))
code_dir = sys.path[0]

print("[CANN-ZhongZhi] work_dir path is [%s]" % (os.getcwd()))
work_dir = os.getcwd()

print("[CANN-ZhongZhi] start run train shell")
# 执行训练脚本
ckpt_path = "/cann-2021-10-21/attention_ocr/attention_ocr/python/inception_v3.ckpt"
shell_cmd = ("bash %s/npu_train.sh %s %s %s %s %s" % (code_dir, code_dir, work_dir, config.data_url, config.train_url, config.modelarts_ckpt_dir))
os.system(shell_cmd)
print("[CANN-ZhongZhi] finish run train shell")




