#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Usage Instructions:
    Scripts with hyperparameters are in experiments/

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.
"""
from npu_bridge.npu_init import *
import csv
import numpy as np
import pickle
import random
import time
import os
import tensorflow as tf

from data_generator import DataGenerator
from poly_generator import PolyDataGenerator
from maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## chip options
flags.DEFINE_string('chip', 'npu', "run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string('platform', 'linux', 'runtime platform, linux or modelarts')
flags.DEFINE_string("obs_dir", '', 'obs result path, not need on gpu and apulis platform')
flags.DEFINE_boolean("profiling", False, "profiling for performance or not")

## Dataset/method options

flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_classes', -1, 'number of classes to train on (-1 for all).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 40000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', .01, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('poly_order', 1, 'order of polynomial to generate')

## Model options
#flags.DEFINE_string('mod', '', 'modifications to original paper. None, split, both')
flags.DEFINE_bool('use_T', True, 'whether or not to use transformation matrix T')
flags.DEFINE_bool('use_M', True, 'whether or not to use mask M')
flags.DEFINE_bool('share_M', True, 'only effective if use_M is true, whether or not to '
                                    'share masks between weights'
                                    'that contribute to the same activation')
flags.DEFINE_float('temp', 1, 'temperature for gumbel-softmax')
flags.DEFINE_float('logit_init', 0, 'initial logit')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('dim_hidden', 40, 'dimension of fc layer')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- use 32 for '
                                        'miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', True, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/omniglot20way', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('debug', False, 'debug mode. uses less data for fast evaluation.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.debug:
        SUMMARY_INTERVAL = PRINT_INTERVAL = 10
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    elif FLAGS.datasource in ['sinusoid', 'polynomial']:
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == 'sinusoid':
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        elif FLAGS.datasource == 'polynomial':
            batch_x, batch_y = data_generator.generate()
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}


        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_loss1,
                                  model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])
        if itr == 0:
            start_time = time.time()
        if itr != 0 and itr % PRINT_INTERVAL == 0:
            end_time = time.time()
            duration = round(end_time - start_time, 3)
            start_time = time.time()
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + 'prelosses ' + str(np.mean(prelosses)) + ', ' + 'postlosses ' + str(np.mean(postlosses)) + ', ' + 'time ' + str(duration) + 's'
            print(print_str)
            #print sess.run(model.total_probs)
            prelosses, postlosses = [], []

        if itr != 0 and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource not in ['sinusoid', 'polynomial']:
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1,
                                     model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1,
                                     model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,
                             model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    if FLAGS.datasource == 'miniimagenet':
        NUM_TEST_POINTS = 4000
    elif FLAGS.datasource == 'polynomial':
        NUM_TEST_POINTS = 20
    else:
        NUM_TEST_POINTS = 600
    for point_n in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {model.meta_lr: 0.0}
        elif FLAGS.datasource == 'sinusoid':
            batch_x, batch_y, amp, phase = data_generator.generate(train=FLAGS.train)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
        elif FLAGS.datasource == 'polynomial':
            batch_x, batch_y = data_generator.generate()
            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

            ########## plotting code
            import matplotlib.pyplot as plt
            from matplotlib import rc
            import matplotlib
            matplotlib.rcParams.update({'font.size': 25})
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10)
            plt.plot(inputa.flatten(), labela.flatten(), 'ro')
            plt.plot(inputb.flatten(), labelb.flatten(), 'r,')
            outputbs = sess.run(model.outputbs, feed_dict)
            plt.plot(inputb.flatten(), outputbs[0].flatten(), color='#bfbfbf', marker=',', linestyle='None')
            plt.plot(inputb.flatten(), outputbs[1].flatten(), color='#666666', marker=',', linestyle='None')
            plt.plot(inputb.flatten(), outputbs[9].flatten(), color='#000000', marker=',', linestyle='None')
            plt.title('Polynomial order ' + str(FLAGS.poly_order))
            plt.legend()
            axes = plt.gca()
            axes.set_xlim([-2, 2])
            axes.set_ylim([-5.1, 5.1])
            plt.savefig(FLAGS.logdir + '/' + exp_string + '/' + str(point_n) + '.png')
            #plt.savefig(str(point_n) + '.png')
            plt.cla()

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:
            result = sess.run([model.total_loss1] + model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))
    filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + \
               '_stepsize' + str(FLAGS.update_lr) + '_testiter' + str(FLAGS.test_iter)
    with open(filename + '.pkl', 'w') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(filename + '.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)


def main():

    if FLAGS.datasource in ['sinusoid', 'polynomial']:
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    elif FLAGS.datasource == 'miniimagenet':
        if FLAGS.train:
            test_num_updates = 1  # eval on at least one update during training
        else:
            test_num_updates = 10
    else:
        test_num_updates = 10

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        #data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
        # Use 10 val samples (meta-SGD, 4.1 paragraph 2 first line)
        data_generator = DataGenerator(FLAGS.update_batch_size+10, FLAGS.meta_batch_size)
    elif FLAGS.datasource == 'polynomial':
        if FLAGS.train:
            data_generator = PolyDataGenerator(FLAGS.update_batch_size+10, FLAGS.meta_batch_size)
        else:
            data_generator = PolyDataGenerator(4000, FLAGS.meta_batch_size)
    elif FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
        assert FLAGS.meta_batch_size == 1
        assert FLAGS.update_batch_size == 1
        data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
    elif FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
        if FLAGS.train:
            data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
        else:
            data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
    else:
        assert FLAGS.datasource == 'omniglot'
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        input_tensors = None
        tf_data_load = False

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=3)

    if FLAGS.chip == 'npu':
        print("************chip is npu************")
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

        ## Performance Profiling
        ## refer to link:https://support.huaweicloud.com/Development-tg-cann202training1/atlasprofilingtrain_16_0003.html
        if FLAGS.profiling:
            work_dir = os.getcwd()
            profiling_dir = os.path.join(work_dir, "npu_profiling")
            if not os.path.exists(profiling_dir):
                os.makedirs(profiling_dir)
            options = '{"output": "%s", "task_trace": "on", "aicpu": "on"}' % (profiling_dir)
            custom_op.parameter_map["profiling_mode"].b = True
            custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)
        sess = tf.InteractiveSession(config=config)
    elif FLAGS.chip == 'gpu':
        print("************chip is gpu************")
        sess = tf.InteractiveSession()
    else:
        raise RuntimeError('************chip [%s] has not supported************' % FLAGS.chip)

    if not FLAGS.train:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+\
                 '.mbs_'+str(FLAGS.meta_batch_size) + \
                 '.ubs_' + str(FLAGS.train_update_batch_size) + \
                 '.numstep' + str(FLAGS.num_updates) + \
                 '.updatelr' + str(FLAGS.train_update_lr) + \
                 '.temp' + str(FLAGS.temp)

    if FLAGS.debug:
        exp_string += '!DEBUG!'

    if FLAGS.use_T and FLAGS.use_M and FLAGS.share_M:
        exp_string += 'MTnet'
    if FLAGS.use_T and not FLAGS.use_M:
        exp_string += 'Tnet'
    if not FLAGS.use_T and FLAGS.use_M and FLAGS.share_M:
        exp_string += 'Mnet'
    if FLAGS.use_T and FLAGS.use_M and not FLAGS.share_M:
        exp_string += 'MTnet_noshare'
    if not FLAGS.use_T and FLAGS.use_M and not FLAGS.share_M:
        exp_string += 'Mnet_noshare'
    if not FLAGS.use_T and not FLAGS.use_M:
        exp_string += 'MAML'

    if FLAGS.datasource == 'polynomial':
        exp_string += 'ord' + str(FLAGS.poly_order)
    if FLAGS.num_train_classes != -1:
        exp_string += 'ntc' + str(FLAGS.num_train_classes)
    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    print (flags.FLAGS.__flags)
    print (exp_string)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

    if FLAGS.platform == 'modelarts':
        from help_modelarts import modelarts_result2obs
        modelarts_result2obs(FLAGS)

if __name__ == "__main__":
    main()

