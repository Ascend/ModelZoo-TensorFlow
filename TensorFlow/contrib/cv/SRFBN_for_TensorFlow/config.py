from npu_bridge.npu_init import *
import os

class config:
    def __init__(self):
        self.batchsize = 1#一次处理的样本数量
        self.Process_num = 3 #进程数量
        self.maxsize = 200 #最大的大小
        self.ngpu = 1 #gpu数量
        self.imagesize = 64#图片大小
        self.scale = 3#缩放规模
        self.epoch = 1000#迭代次数
        #创建检查点，记录，结果目录
        self.checkpoint_dir = "./model"#检查点目录
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.log_dir = "./log"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.result = "./result"
        if not os.path.exists(self.result):
            os.mkdir(self.result)



class SRFBN_config(config):
    def __init__(self):
        super(SRFBN_config, self).__init__()
        self.istrain = True#正在训练还是正在测试
        self.istest = not self.istrain
        self.c_dim = 3 #color channel 可以训练灰度图也可以训练RGB图
        self.in_channels = 3
        self.out_channels = 3
        self.num_features = 32#base number of filter
        self.num_steps = 4#时间步
        self.num_groups = 6#FBB中feedbackblock中projection group数量
        self.BN = True#
        if self.BN:
            self.BN_type = "BN" # "BN" # or "IN"
        self.act_type = "prelu" #activation function
        self.loss_type = "L2"
        self.lr_steps = [150, 300, 550, 750]#迭代次数表
        self.lr_gama = 1#参数
        self.learning_rate = 2e-7#学习率
        self.load_premodel = True 
        #创建目录
        self.srfbn_logdir = "%s/srfbn" % self.log_dir
        if not os.path.exists(self.srfbn_logdir):
            os.mkdir(self.srfbn_logdir)
        self.srfbn_result = "%s/srfbn" % self.result
        if not os.path.exists(self.srfbn_result):
            os.mkdir(self.srfbn_result)

