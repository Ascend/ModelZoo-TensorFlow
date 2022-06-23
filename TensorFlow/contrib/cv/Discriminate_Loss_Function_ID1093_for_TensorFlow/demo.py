# coding=utf-8
from npu_bridge.npu_init import *
import os
import argparse
import datetime
import moxing as mox

## Code dir: /home/work/user-job-dir/code # ��ModelArts�ϵĴ���洢Ŀ¼����Ŀ¼���ᱻ������Ϊcode����
## Work dir: /home/work/workspace/device2 # device id��job����
print("===>>>{}".format(os.getcwd()))
print(os.system('env'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str, default="../output")  # PyCharm�������� OBS Path·��
    parser.add_argument("--data_url", type=str, default="../dataset")  # PyCharm�������� Data Path in OBS·��
    config = parser.parse_args()

    # Ϊ�˷��㣬���ǻὫѵ��ʱ�õ����ݼ����������� /cache Ŀ¼�£�ImageNet TFRecord��ʽ148G��ҪԼ280s����Ϊ��������С�ļ����ݼ�������Ϊtar�ļ��󿽱���ȥ�ٽ�ѹ
    # copy dataset from obs to local
    # dataset will be saved under /cache/ilsvrc2012_tfrecord while the results will be saved under /cache/results
    local_dir = '/cache/ilsvrc2012_tfrecord'
    start = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to local dir:{}".format(config.data_url, local_dir))
    mox.file.copy_parallel(src_url=config.data_url, dst_url=local_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from obs to local, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))

    # ��ʼѵ���ű�������ֻ��Ҫ��ѵ���ű��е�dataset pathĬ��ָ��Ϊ/cache�е���Ӧ���ݼ�·�����ɣ�ͬʱѵ����log, snapshot���ļ�Ҳ����д��/cache�µ�ĳһ�ض��ļ��У����ع�̬д���ÿ�η���obsҪ�죬����Ҫ�ڴ��������mox��ȱ���������ֶ�kill������Ͳ��ᱣ���м�����������Զ�ʱcopyһ�¡�
    #  run training
    print("===>>>Begin training:")
    os.system('bash /home/work/user-job-dir/code/run_1p.sh')  # ��ʾ���ľ���ѵ���ű�Ϊrun_1p.sh
    print("===>>>Training finished:")

    # ���ѵ����������Ҫ�������м���������obs��Ŀ��obs·��Ϊ����֮ǰ�����--train_url
    #  copy results from local to obs
    local_dir = '/cache/result'
    remote_dir = os.path.join(config.train_url, 'result')
    if not mox.file.exists(remote_dir):
        mox.file.make_dirs(remote_dir)
    start = datetime.datetime.now()
    print("===>>>Copy files from local dir:{} to obs:{}".format(local_dir, remote_dir))
    mox.file.copy_parallel(src_url=local_dir, dst_url=remote_dir)
    end = datetime.datetime.now()
    print("===>>>Copy from local to obs, time use:{}(s)".format((end - start).seconds))
    files = os.listdir(local_dir)
    print("===>>>Files number:", len(files))
