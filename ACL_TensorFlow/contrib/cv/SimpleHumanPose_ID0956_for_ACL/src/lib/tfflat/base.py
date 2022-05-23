from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from collections import OrderedDict as dict
import setproctitle
import os
import os.path as osp
import glob
import abc
import math

from .net_utils import average_gradients, aggregate_batch, get_optimizer, get_tower_summary_dict
from .saver import load_model, Saver
from .timer import Timer
from .logger import colorlogger
from .utils import approx_equal


class ModelDesc(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._loss = None
        self._inputs = []
        self._outputs = []
        self._tower_summary = []

    def set_inputs(self, *vars):
        self._inputs = vars

    def set_outputs(self, *vars):
        self._outputs = vars

    def set_loss(self, var):
        if not isinstance(var, tf.Tensor):
            raise ValueError("Loss must be an single tensor.")
        # assert var.get_shape() == [], 'Loss tensor must be a scalar shape but got {} shape'.format(var.get_shape())
        self._loss = var

    def get_loss(self, include_wd=False):
        if self._loss is None:
            raise ValueError("Network doesn't define the final loss")

        if include_wd:
            weight_decay = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            weight_decay = tf.add_n(weight_decay)
            return self._loss + weight_decay
        else:
            return self._loss

    def get_inputs(self):
        if len(self._inputs) == 0:
            raise ValueError("Network doesn't define the inputs")
        return self._inputs

    def get_outputs(self):
        if len(self._outputs) == 0:
            raise ValueError("Network doesn't define the outputs")
        return self._outputs

    def add_tower_summary(self, name, vars, reduced_method='mean'):
        assert reduced_method == 'mean' or reduced_method == 'sum', \
            "Summary tensor only supports sum- or mean- reduced method"
        if isinstance(vars, list):
            for v in vars:
                if vars.get_shape() == None:
                    print('Summary tensor {} got an unknown shape.'.format(name))
                else:
                    assert v.get_shape().as_list() == [], \
                        "Summary tensor only supports scalar but got {}".format(v.get_shape().as_list())
                tf.add_to_collection(name, v)
        else:
            if vars.get_shape() == None:
                print('Summary tensor {} got an unknown shape.'.format(name))
            else:
                assert vars.get_shape().as_list() == [], \
                    "Summary tensor only supports scalar but got {}".format(vars.get_shape().as_list())
            tf.add_to_collection(name, vars)
        self._tower_summary.append([name, reduced_method])

    @abc.abstractmethod
    def make_network(self, is_train):
        pass


class Base(object):
    __metaclass__ = abc.ABCMeta
    """
    build graph:
        _make_graph
            make_inputs
            make_network
                add_tower_summary
        get_summary
    
    train/test
    """

    def __init__(self, net, cfg, data_iter=None, log_name='logs.txt'):
        self._input_list = []
        self._output_list = []
        self._outputs = []
        self.graph_ops = None

        self.net = net
        self.cfg = cfg

        self.cur_epoch = 0

        self.summary_dict = {}

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

        # initialize tensorflow
        tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=npu_config_proto(config_proto=tfconfig))

        # build_graph
        self.build_graph()

        # get data iter
        self._data_iter = data_iter

    @abc.abstractmethod
    def _make_data(self):
        return

    @abc.abstractmethod
    def _make_graph(self):
        return

    def build_graph(self):
        # all variables should be in the same graph and stored in cpu.
        with tf.device('/cpu:0'):
            tf.set_random_seed(2333)
            self.graph_ops = self._make_graph()
            if not isinstance(self.graph_ops, list) and not isinstance(self.graph_ops, tuple):
                self.graph_ops = [self.graph_ops]
        self.summary_dict.update(get_tower_summary_dict(self.net._tower_summary))

    def load_weights(self, model=None):

        if model == 'last_epoch':
            sfiles = os.path.join(self.cfg.model_dump_dir, 'snapshot_*.ckpt.meta')
            sfiles = glob.glob(sfiles)
            if len(sfiles) > 0:
                sfiles.sort(key=os.path.getmtime)
                sfiles = [i[:-5] for i in sfiles if i.endswith('.meta')]
                model = sfiles[-1]
            else:
                self.logger.critical('No snapshot model exists.')
                return

        if isinstance(model, int):
            model = os.path.join(self.cfg.model_dump_dir_test, 'snapshot_%d.ckpt' % model)  # 修改

        if isinstance(model, str) and (osp.exists(model + '.meta') or osp.exists(model)):
            self.logger.info('Initialized model weights from {} ...'.format(model))
            load_model(self.sess, model)
            if model.split('/')[-1].startswith('snapshot_'):
                self.cur_epoch = int(model[model.find('snapshot_') + 9:model.find('.ckpt')])
                self.logger.info('Current epoch is %d.' % self.cur_epoch)
        else:
            self.logger.critical('Load nothing. There is no model in path {}.'.format(model))

    def next_feed(self):
        if self._data_iter is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        for inputs in self._input_list:
            # print("inputs:", self.sess.run(inputs))
            blobs = next(self._data_iter)
            for i, inp in enumerate(inputs):
                inp_shape = inp.get_shape().as_list()
                if None in inp_shape:
                    feed_dict[inp] = blobs[i]
                else:
                    feed_dict[inp] = blobs[i].reshape(*inp_shape)
        return feed_dict


class Trainer(Base):
    def __init__(self, net, cfg, data_iter=None):
        self.lr_eval = cfg.lr
        self.lr = tf.Variable(cfg.lr, trainable=False)
        self._optimizer = get_optimizer(self.lr, cfg.optimizer)

        super(Trainer, self).__init__(net, cfg, data_iter, log_name='train_logs.txt')

        # make data
        self._data_iter, self.itr_per_epoch = self._make_data()

    def _make_data(self):
        from dataset import Dataset
        from gen_batch import generate_batch

        d = Dataset()
        train_data = d.load_train_data()

        from tfflat.data_provider import DataFromList, MultiProcessMapDataZMQ, BatchData, MapData
        data_load_thread = DataFromList(train_data)
        if self.cfg.multi_thread_enable:
            data_load_thread = MultiProcessMapDataZMQ(data_load_thread, self.cfg.num_thread, generate_batch,
                                                      strict=True)
        else:
            data_load_thread = MapData(data_load_thread, generate_batch)
        data_load_thread = BatchData(data_load_thread, self.cfg.batch_size)

        data_load_thread.reset_state()
        dataiter = data_load_thread.get_data()

        return dataiter, math.ceil(len(train_data) / self.cfg.batch_size / self.cfg.num_gpus)

    def _make_graph(self):
        self.logger.info("Generating training graph on {} GPUs ...".format(self.cfg.num_gpus))

        weights_initializer = slim.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.)
        biases_regularizer = tf.no_regularizer
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.cfg.weight_decay)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/cpu:0'):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        # Force all Variables to reside on the CPU.
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                 slim.conv2d_transpose, slim.separable_conv2d,
                                                 slim.fully_connected],
                                                weights_regularizer=weights_regularizer,
                                                biases_regularizer=biases_regularizer,
                                                weights_initializer=weights_initializer,
                                                biases_initializer=biases_initializer):
                                # loss over single GPU
                                self.net.make_network(is_train=True)
                                if i == self.cfg.num_gpus - 1:
                                    loss = self.net.get_loss(include_wd=True)
                                else:
                                    loss = self.net.get_loss()
                                self._input_list.append(self.net.get_inputs())

                        tf.get_variable_scope().reuse_variables()

                        if i == 0:
                            if self.cfg.num_gpus > 1 and self.cfg.bn_train is True:
                                self.logger.warning("BN is calculated only on single GPU.")
                            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                            with tf.control_dependencies(extra_update_ops):
                                grads = self._optimizer.compute_gradients(loss)
                        else:
                            grads = self._optimizer.compute_gradients(loss)
                        final_grads = []
                        with tf.variable_scope('Gradient_Mult') as scope:
                            for grad, var in grads:
                                final_grads.append((grad, var))
                        tower_grads.append(final_grads)

        if len(tower_grads) > 1:
            grads = average_gradients(tower_grads)
        else:
            grads = tower_grads[0]

        apply_gradient_op = self._optimizer.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op, *extra_update_ops)

        return train_op

    def train(self):

        # saver
        self.logger.info('Initialize saver ...')
        train_saver = Saver(self.sess, tf.global_variables(), self.cfg.model_dump_dir)

        # initialize weights
        self.logger.info('Initialize all variables ...')
        self.sess.run(tf.variables_initializer(tf.global_variables(), name='init'))
        self.load_weights('last_epoch' if self.cfg.continue_train else self.cfg.init_model)

        self.logger.info('Start training ...')
        start_itr = self.cur_epoch * self.itr_per_epoch + 1
        end_itr = self.itr_per_epoch * self.cfg.end_epoch + 1
        for itr in range(start_itr, end_itr):
            self.tot_timer.tic()

            self.cur_epoch = itr // self.itr_per_epoch
            setproctitle.setproctitle('train epoch:' + str(self.cur_epoch))

            # apply current learning policy
            cur_lr = self.cfg.get_lr(self.cur_epoch)
            if not approx_equal(cur_lr, self.lr_eval):
                print(self.lr_eval, cur_lr)
                self.sess.run(tf.assign(self.lr, cur_lr))

            # input data
            self.read_timer.tic()
            feed_dict = self.next_feed()
            self.read_timer.toc()

            # train one step
            self.gpu_timer.tic()
            _, self.lr_eval, *summary_res = self.sess.run(
                [self.graph_ops[0], self.lr, *self.summary_dict.values()], feed_dict=feed_dict)
            self.gpu_timer.toc()

            itr_summary = dict()
            for i, k in enumerate(self.summary_dict.keys()):
                itr_summary[k] = summary_res[i]

            screen = [
                'Epoch %d itr %d/%d:' % (self.cur_epoch, itr, self.itr_per_epoch),
                'lr: %g' % (self.lr_eval),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    self.tot_timer.average_time, self.gpu_timer.average_time, self.read_timer.average_time),
                '%.2fh/epoch' % (self.tot_timer.average_time / 3600. * self.itr_per_epoch),
                ' '.join(map(lambda x: '%s: %.4f' % (x[0], x[1]), itr_summary.items())),
            ]

            # TODO(display stall?)
            if itr % self.cfg.display == 0:
                self.logger.info(' '.join(screen))

            if itr % self.itr_per_epoch == 0:
                train_saver.save_model(self.cur_epoch)

            self.tot_timer.toc()


class Tester(Base):
    def __init__(self, net, cfg, data_iter=None):
        super(Tester, self).__init__(net, cfg, data_iter, log_name='test_logs.txt')

    def next_feed(self, batch_data=None):
        if self._data_iter is None and batch_data is None:
            raise ValueError('No input data.')
        feed_dict = dict()
        if batch_data is None:
            for inputs in self._input_list:
                blobs = next(self._data_iter)
                for i, inp in enumerate(inputs):
                    inp_shape = inp.get_shape().as_list()
                    if None in inp_shape:
                        feed_dict[inp] = blobs[i]
                    else:
                        feed_dict[inp] = blobs[i].reshape(*inp_shape)
        else:
            assert isinstance(batch_data, list) or isinstance(batch_data, tuple), "Input data should be list-type."
            assert len(batch_data) == len(self._input_list[0]), "Input data is incomplete."

            batch_size = self.cfg.batch_size
            if self._input_list[0][0].get_shape().as_list()[0] is None:
                # fill batch
                for i in range(len(batch_data)):
                    batch_size = (len(batch_data[i]) + self.cfg.num_gpus - 1) // self.cfg.num_gpus
                    total_batches = batch_size * self.cfg.num_gpus
                    left_batches = total_batches - len(batch_data[i])
                    if left_batches > 0:
                        batch_data[i] = np.append(batch_data[i], np.zeros((left_batches, *batch_data[i].shape[1:])),
                                                  axis=0)
                        self.logger.warning("Fill some blanks to fit batch_size which wastes %d%% computation" % (
                                left_batches * 100. / total_batches))
            else:
                assert self.cfg.batch_size * self.cfg.num_gpus == len(batch_data[0]), \
                    "Input batch doesn't fit placeholder batch."

            for j, inputs in enumerate(self._input_list):
                for i, inp in enumerate(inputs):
                    feed_dict[inp] = batch_data[i][j * batch_size: (j + 1) * batch_size]

            # @TODO(delete)
            assert (j + 1) * batch_size == len(batch_data[0]), 'check batch'
        return feed_dict, batch_size

    def _make_graph(self):
        self.logger.info("Generating testing graph on {} GPUs ...".format(self.cfg.num_gpus))

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(self.cfg.num_gpus):
                with tf.device('/cpu:0'):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with slim.arg_scope([slim.model_variable, slim.variable], device='/device:CPU:0'):
                            self.net.make_network(is_train=False)
                            self._input_list.append(self.net.get_inputs())
                            self._output_list.append(self.net.get_outputs())

                        tf.get_variable_scope().reuse_variables()

        self._outputs = aggregate_batch(self._output_list)

        # run_meta = tf.RunMetadata()
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)
        #
        # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        # params = tf.profiler.profile(self.sess.graph, run_meta=run_meta, cmd='op', options=opts)

        # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        # from IPython import embed; embed()

        return self._outputs

    def predict_one(self, data=None):
        # TODO(reduce data in limited batch)
        assert len(self.summary_dict) == 0, "still not support scalar summary in testing stage"
        setproctitle.setproctitle('test epoch:' + str(self.cur_epoch))

        self.read_timer.tic()
        feed_dict, batch_size = self.next_feed(data)
        self.read_timer.toc()

        self.gpu_timer.tic()
        res = self.sess.run([*self.graph_ops, *self.summary_dict.values()], feed_dict=feed_dict)
        self.gpu_timer.toc()

        if data is not None and len(data[0]) < self.cfg.num_gpus * batch_size:
            for i in range(len(res)):
                res[i] = res[i][:len(data[0])]

        return res

    def test(self):
        pass
