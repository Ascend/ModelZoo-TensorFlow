from npu_bridge.npu_init import *
import tensorflow as tf
import  os
from Basic_Model import basic_network


class SRFBN(basic_network):
    def __init__(self, sess, cfg):
        super(SRFBN, self).__init__(cfg)
        self.sess = sess
        imageshape = [cfg.batchsize, cfg.imagesize, cfg.imagesize, cfg.c_dim]
        labelshape = [cfg.batchsize, cfg.imagesize * cfg.scale, cfg.imagesize * cfg.scale, cfg.c_dim]
        self.imageplaceholder = tf.placeholder(dtype=tf.float32, shape=imageshape, name="image")
        self.labelplaceholder = tf.placeholder(dtype=tf.float32, shape=labelshape, name="label")
        self.last_hidden = None
        self.should_reset = True
        self.outs = []
        #FB块
    def FeedBackBlock(self, x, num_features, num_groups, act_type, name="FBB"):
        if self.cfg.scale == 1:
            stride = 1
            padding = "SAME"
            kernel_size = 5
        if self.cfg.scale==2:
            stride = 2
            padding = "SAME"
            kernel_size = 6
        if self.cfg.scale == 3:
            stride = 3
            padding = "SAME"
            kernel_size = 7
        if self.cfg.scale == 4:
            stride = 4
            padding = "SAME"
            kernel_size = 8
        if self.should_reset:
            self.last_hidden = x
            self.should_reset = False
        x = tf.concat([x, self.last_hidden], 3)
        x = self.ConvBlock(x, 2*num_features, num_features, kernel_size=1, name="FeedBack_compress_in",
                           act_type=act_type)

        lr_features = []
        hr_features = []
        lr_features.append(x)
        for i in range(num_groups):
            x = tf.concat(lr_features, 3)
            if i > 0:
                x = self.ConvBlock(x, num_features*(i+1), num_features, kernel_size=1,stride=1,
                                   padding=padding, act_type=act_type, name="%s_%d"%(name, i))
            x = self.DeConvBlock(x, num_features, num_features, kernel_size=kernel_size, stride=stride,
                                 padding=padding, act_type=act_type, name="%s_%d"%(name, i))
            hr_features.append(x)
            x = tf.concat(hr_features, 3)
            if i > 0:
                x = self.ConvBlock(x, num_features*(i+1), num_features, kernel_size=1, stride=1,
                                   padding=padding, act_type=act_type, name="%s_%d"%(name, i))
            x = self.ConvBlock(x, num_features, num_features, kernel_size=kernel_size, stride=stride,
                               padding=padding, act_type=act_type, name="%s_%d"%(name, i))
            lr_features.append(x)
        del hr_features

        x = tf.concat(lr_features[1:], 3)

        x = self.ConvBlock(x, num_features*num_groups, num_features, kernel_size=1,
                           act_type=act_type, name="FeedBack_compress_out")

        self.last_hidden = x

        return x

    def build(self):
        if self.cfg.scale == 2:
            stride = 2
            padding = "SAME"
            kernel_size = 6
        if self.cfg.scale == 3:
            stride = 3
            padding = "SAME"
            kernel_size = 7
        if self.cfg.scale == 4:
            stride = 4
            padding = "SAME"
            kernel_size = 8
        # x = self.sub_mean(self.imageplaceholder) # 暂且当作归一化

        _, height, width, _ = self.imageplaceholder.get_shape().as_list()

        inter_size = tf.constant([height*self.cfg.scale, width*self.cfg.scale])
        inter_res = tf.image.resize_images(self.imageplaceholder, inter_size)
        # inter_res = self.imageplaceholder

        x = self.ConvBlock(self.imageplaceholder, self.cfg.in_channels, 4 * self.cfg.num_features, kernel_size=3,
                           act_type=self.cfg.act_type, padding="SAME", name="conv_in")
        x = self.ConvBlock(x, 4*self.cfg.num_features, self.cfg.num_features, kernel_size=1,
                           act_type=self.cfg.act_type, padding="SAME", name="feat_in")
        # outs = []
        for i in range(self.cfg.num_steps):
            if i == 0:
                self.should_reset=True
            t = self.FeedBackBlock(x, self.cfg.num_features, self.cfg.num_groups, self.cfg.act_type, name="FBB_%d"%i)
            t = self.DeConvBlock(t, self.cfg.num_features, self.cfg.num_features, kernel_size=kernel_size,
                                 stride=stride, padding=padding, act_type="relu", name="out_%d"%i)
            t = self.ConvBlock(t, self.cfg.num_features, self.cfg.out_channels, kernel_size=3, stride=1,
                               act_type="tanh", padding="SAME", name="conv_out")
            t = inter_res + t
            t = tf.clip_by_value(t, -1.0, 1.0)
            # t = t + inter_res
            # t = self.add_mean(t)
            self.outs.append(t)
            #训练步骤
    def train_step(self):
        self.build()
        print("This Net has Params num is %f MB" % (self.params_count * 4 / 1024 / 1024))  # float32
        tf.summary.image("image/HR", self.labelplaceholder, max_outputs=1)
        out = tf.add_n(self.outs)/self.cfg.num_steps

        tf.summary.image("image/SR", out, max_outputs=1)
        tf.summary.image("image/LR", self.imageplaceholder, max_outputs=1)

        self.l2_regularization_loss = tf.reduce_sum(tf.get_collection("weights_l2_loss"))

        self.losses = [self.calc_loss(x=x, y=self.labelplaceholder, loss_type=self.cfg.loss_type) for x in self.outs]
        self.losses = tf.reduce_sum(self.losses)/len(self.losses)/self.cfg.batchsize + self.l2_regularization_loss

        tf.summary.scalar('loss/total', self.losses)
        tf.summary.scalar('loss/l2_loss', self.l2_regularization_loss)

        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        #加载检查点
    def load(self):
        model_name = "SRFBN.model"
        model_dir = "%s_%s_%s_%s_c%d_x%s" % (
        "SRFBN", self.cfg.num_features, self.cfg.num_steps, self.cfg.num_groups, self.cfg.c_dim, self.cfg.scale)
        checkpoint_dir = os.path.join(self.cfg.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
            step = int(os.path.basename(ckpt_path).split('-')[1])
            print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            print("\nCheckpoint Loading Failed! \n")

        return step
    #保存当前模型
    def save(self, step):
        model_name = "SRFBN.model"
        model_dir = "%s_%s_%s_%s_c%d_x%s" % \
                    ("SRFBN", self.cfg.num_features, self.cfg.num_steps,
                     self.cfg.num_groups, self.cfg.c_dim, self.cfg.scale)
        checkpoint_dir = os.path.join(self.cfg.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        #测试
    def test(self, width, height):
        self.cfg.batchsize = 1
        testshape = [self.cfg.batchsize, height, width, self.cfg.c_dim]
        labelshape = [self.cfg.batchsize, height*self.cfg.scale, width*self.cfg.scale, self.cfg.c_dim]
        self.imageplaceholder = tf.placeholder(dtype=tf.float32, shape=testshape)
        self.labelplaceholder = tf.placeholder(dtype=tf.float32, shape=labelshape)
        self.build()
        # self.outs = [self.add_mean(x) for x in self.outs]
        out = tf.add_n(self.outs)/self.cfg.num_steps
        # out = tf.concat(self.outs, -1)
        return out


if __name__ == '__main__':
    from config import SRFBN_config as config
    cfg = config()
    sess = tf.Session(config=npu_config_proto())
    net = SRFBN(sess, cfg)
    train_step = net.train_step()
