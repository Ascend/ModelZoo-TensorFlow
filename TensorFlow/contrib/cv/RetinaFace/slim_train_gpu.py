import argparse
import os
import sys
import time

import tensorflow as tf
from absl import logging
slim = tf.contrib.slim

from modules.anchor import prior_box
from modules.losses import MultiBoxLoss
# from modules.losses_npu_mod import MultiBoxLoss
# from modules.loss_npu import MultiBoxLoss
from modules.lr_scheduler import MultiStepWarmUpLR
from modules.retina_slim import RetinaFaceModel, test_gn
from modules.utils import (ProgressBar, load_dataset, load_yaml,
                           set_memory_growth, load_sess)
# for dump feature map
from tensorflow.python import debug as tf_debug



def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


class Configs(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        # Global set
        parser.add_argument('--model', type=str, default='res50')
        parser.add_argument('--gpu', type=eval, default=(3,4,5,6))
        # parser.add_argument('--gpu', type=eval, default=(0,))
        parser.add_argument('--log_level', type=str, default='3')
        parser.add_argument('--pretrain_path', type=str, default='pretrain_models')
        self.args = parser.parse_args()


class warm_up_lr(object):
    def __init__(self, initial_learning_rate, lr_steps, lr_rate, warmup_steps=0., min_lr=0.):
        assert warmup_steps <= lr_steps[0]
        assert min_lr <= initial_learning_rate
        self.lr_steps_value = [initial_learning_rate]
        self.lr_steps = lr_steps
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        for _ in range(len(lr_steps)):
            self.lr_steps_value.append(self.lr_steps_value[-1] * lr_rate)

    def __call__(self, step):
        if step <= self.warmup_steps:
            lr = self.min_lr + step * \
                (self.lr_steps_value[0] - self.min_lr) / self.warmup_steps
        else:
            if step >= self.lr_steps[-1]:
                lr = self.lr_steps_value[-1]
            else:
                for i in range(len(self.lr_steps)):
                    if step < self.lr_steps[i]:
                        lr = self.lr_steps_value[i]
                        break
        return lr


def main(args):
    root_path, _ = os.path.split(os.path.abspath(__file__))
    ######################## init ########################
    res_config = os.path.join(root_path, "configs/retinaface_res50.yaml")
    mbn_config = os.path.join(root_path, "configs/retinaface_mbv2.yaml")
    config_path = res_config if args.model == 'res50' else mbn_config
    model_ckpt_name = 'resnet_v1_50.ckpt' if args.model == 'res50' else None
    visible_devices = ''
    for i in args.gpu:
        visible_devices += '{}, '.format(i)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.log_level
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    # set_memory_growth()
    # set log
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    # get cfg
    cfg = load_yaml(config_path)
    cfg.update({'root_path': root_path})
    # set checkpoint path
    pretrain_path = os.path.join(root_path, args.pretrain_path)
    pretrain_model_ckpt = os.path.join(pretrain_path, model_ckpt_name) if model_ckpt_name is not None else None
    checkpoint_dir = os.path.join(root_path, 'checkpoints', cfg['sub_name'])
    slide_steps = cfg['steps']
    base_anchor_size = cfg['basesize']
    input_size = cfg['input_size']
    batch_size = cfg['batch_size']
    scale_num = 3
    _base_sacle = 2.0**(1.0 / 3)
    five_step_min_sizes = [
        [base_anchor_size*(_base_sacle**(i+j*scale_num)) for i in range(scale_num)] for j in range(len(slide_steps))]
    cfg['min_sizes'] = cfg['min_sizes'] if len(
        slide_steps) == 3 else five_step_min_sizes
    anchor_num = sum([(input_size/i)**2*scale_num for i in slide_steps])
    # define prior box
    priors = prior_box((input_size, input_size), five_step_min_sizes,  slide_steps, cfg['clip'])
    # load dataset
    train_dataset = load_dataset(cfg, priors, shuffle=True)
    # define network
    model = RetinaFaceModel(cfg, training=True)
    # define losses function
    multi_box_loss = MultiBoxLoss()
    # define optimizer
    steps_per_epoch = cfg['dataset_len'] // (batch_size*len(args.gpu))
    learning_rate = warm_up_lr(
        initial_learning_rate=cfg['init_lr'],
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
        lr_rate=cfg['lr_rate'],
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
        min_lr=cfg['min_lr'])

    ######################## build graph ########################
    lrholder = tf.placeholder(tf.float32, [])
    optimizer = tf.train.MomentumOptimizer(learning_rate=lrholder,
                                           momentum=0.9,
                                           use_nesterov=True)
    # creat data iteration
    data_iteration = train_dataset.make_initializable_iterator()
    inputs, labels = data_iteration.get_next()
    # print(labels)

    # if use multi gpu, the use with device
    all_grads, grads_, var_, checker, losses = [], [], [], [], {}

    # test_func = test_gn(training=True)
    for idx, g in enumerate(['/gpu:'+str(i) for i in range(len(args.gpu))]):
        with tf.device(g):
            # get loss
            predictions = model(inputs)
            # with tf.device('/cpu:0'):
            losses['loc'], losses['landm'], losses['class'] = multi_box_loss(labels, predictions)
            losses['reg'] = tf.math.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            w_loss = [losses['reg'], losses['class'], 0.5*losses['loc'], 0.2*losses['landm']]
            total_loss = tf.add_n(w_loss)
            # for test layer
            # total_loss = test_func(tf.truncated_normal(inputs.shape))

            # get gradient
            if idx == 0:
                ops = tf.get_default_graph().get_operations()
                _ = [
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, x) for x in ops
                    if ("AssignMovingAvg" in x.name and x.type == "AssignSub")
                ]
                upds = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            tf.get_variable_scope().reuse_variables()
            vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads = optimizer.compute_gradients(total_loss, vars_list)
            all_grads.append(grads)

    for i in zip(*all_grads):
        # for tmp in i:
        #     if None in tmp:
        #         print(i)
        #         input()
        #         break
        tmp_grads = [_[0] for _ in i]
        var_.append(i[0][1])
        grads_.append(sum(tmp_grads) / len(tmp_grads))

    grad_var = list(zip(grads_, var_))
    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
        with tf.control_dependencies(upds):
            train_op = optimizer.apply_gradients(grad_var)
    saver = tf.train.Saver()
    ######################## session train ########################
    with tf.Session() as sess:
        latest_ckp = tf.train.latest_checkpoint(checkpoint_dir)
        step = 0 if not latest_ckp else int(
            latest_ckp.split('.')[0].split('-')[-1])
        # session init
        sess.run(tf.initialize_all_variables())
        sess.run(data_iteration.initializer)
        # dump setting
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")

        if step == 0 and pretrain_model_ckpt is not None:
            # this only for res50 pretrain
            pre_var_list = slim.get_variables_to_restore(include=['resnet_v1_50'])
            load_sess(checkpoint_dir,
                      sess=sess,
                      init=True,
                      modpath=pretrain_model_ckpt,
                      var_list=pre_var_list)
        else:
            load_sess(checkpoint_dir, sess=sess, init=True)
        prog_bar = ProgressBar(steps_per_epoch, step % steps_per_epoch)
        remain_steps = max(steps_per_epoch * cfg['epoch'] - step, 0)

        for _ in range(remain_steps):

            step += 1
            lr = learning_rate(step)
            float_total_loss, _ = sess.run(
                [total_loss, train_op], feed_dict={lrholder: lr})

            prog_bar.update("epoch={}/{}, loss={:.4f}, lr={:.1e}".format(
                ((step - 1) // steps_per_epoch) + 1, cfg['epoch'], float_total_loss, lr))
            if step % cfg['save_steps'] == 0:
                saver.save(sess, checkpoint_dir +'/retinaface', global_step=step)
                print("\n[*] save ckpt file at {}".format(
                    tf.train.latest_checkpoint(checkpoint_dir)))

        saver.save(sess, checkpoint_dir +'/retinaface', global_step=step)
        print("\n[*] save ckpt file at {}".format(
            tf.train.latest_checkpoint(checkpoint_dir)))


if __name__ == '__main__':
    all_configs = Configs()
    main(all_configs.args)
