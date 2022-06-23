# coding=utf-8
from npu_bridge.npu_init import *
import os
import argparse
import datetime
#import moxing as mox

## Code dir: /home/work/user-job-dir/code # 在ModelArts上的代码存储目录（父目录均会被重命名为code）。
## Work dir: /home/work/workspace/device2 # device id因job而异
print("===>>>{}".format(os.getcwd()))
print(os.system('env'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="../output")  # PyCharm插件传入的 OBS Path路径
    parser.add_argument("--data_url", type=str, default="../dataset")  # PyCharm插件传入的 Data Path in OBS路径
    config = parser.parse_args()

    # 为了方便，我们会将训练时用的数据集拷贝到本地 /cache 目录下，ImageNet TFRecord格式148G需要约280s，如为其他大量小文件数据集建议打包为tar文件后拷贝过去再解压
    # copy dataset from obs to local
    # dataset will be saved under /cache/ilsvrc2012_tfrecord while the results will be saved under /cache/results
    local_dir = '/cache/ilsvrc2012_tfrecord'
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to local dir:{}".format(config.data_url, local_dir))
    #mox.file.copy_parallel(src_url=config.data_url, dst_url=local_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to local, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))

    # 开始训练脚本，我们只需要将训练脚本中的dataset path默认指定为/cache中的相应数据集路径即可；同时训练的log, snapshot等文件也可以写入/cache下的某一特定文件夹，本地固态写入比每次访问obs要快，不需要在代码里调用mox，缺点就是如果手动kill掉任务就不会保留中间结果，建议可以定时copy一下。
    #  run training
    print("===>>>Begin training:")
    os.system('bash /home/work/user-job-dir/code/run_1p.sh')  # 本示例的具体训练脚本为run_1p.sh
    print("===>>>Training finished:")

    # 完成训练后将我们需要保留的中间结果拷贝到obs，目的obs路径为我们之前传入的--train_url
    #  copy results from local to obs
    local_dir = '/cache/result'
    remote_dir = os.path.join(config.train_url, 'result')
    #if not mox.file.exists(remote_dir):
        #mox.file.make_dirs(remote_dir)
    start = datetime.datetime.now()
    print("===>>>Copy files from local dir:{} to obs:{}".format(local_dir, remote_dir))
    #mox.file.copy_parallel(src_url=local_dir, dst_url=remote_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from local to obs, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))
