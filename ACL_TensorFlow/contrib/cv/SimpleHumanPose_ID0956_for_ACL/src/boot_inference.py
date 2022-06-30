# coding=utf-8
import os
import argparse
import datetime
import moxing as mox

import tensorflow as tf
import os
import sys

# Code dir: /home/work/user-job-dir/code # 在ModelArts上的代码存储目录（父目录均会被重命名为code）。
# Work dir: /home/work/workspace/device2 # device id因job而异

# print(os.system('env'))

if __name__ == '__main__':

    code_dir = os.path.dirname(os.path.realpath(__file__))
    work_dir = os.getcwd()
    print("===>>>code_dir:{}".format(code_dir))
    print("===>>>work_dir{}".format(work_dir))

    parser = argparse.ArgumentParser()

    # 解析输入参数data_url
    parser.add_argument("--data_url", type=str, default="./dataset")  # 这里要写绝对路径吗
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument('--test_epoch', type=str, default='140')
    args = parser.parse_args()

    # 打印config参数
    print("--------config----------")
    for k in list(vars(args).keys()):
        print("key:{}: value:{}".format(k, vars(args)[k]))
    print("--------config----------")

    # 在modelarts创建数据存放目录
    data_dir = "cache/dataset"
    # os.makedirs(data_dir)

    # OBS数据拷贝到modelarts容器内
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(args.data_url, data_dir))
    mox.file.copy_parallel(args.data_url, data_dir)
    files = os.listdir(data_dir)
    print("===>>>Files:", files)

    print("===>>>Begin booting:")
    os.system("python /home/ma-user/modelarts/user-job-dir/code/main/pb_inference.py")
    print("===>>>Test finished:")

    # 训练完成, 结果上传到OBS
    output_dir = "cache/result"
    remote_dir = os.path.join(args.train_url)
    if not mox.file.exists(remote_dir):
        mox.file.make_dirs(remote_dir)
    # start = datetime.datetime.now()
    print("===>>>Copy files from local dir:{} to obs:{}".format(output_dir, remote_dir))
    mox.file.copy_parallel(src_url=output_dir, dst_url=remote_dir)
    # end = datetime.datetime.now()
    # print("===>>>Copy from local to obs, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(output_dir)
    print("===>>>Files number:", len(files))
