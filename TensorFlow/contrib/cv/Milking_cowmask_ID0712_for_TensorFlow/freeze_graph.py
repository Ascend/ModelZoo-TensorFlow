# coding=utf-8
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
from npu_bridge.npu_init import *
import logging
import ast
from absl import flags,app
import tensorflow as tf
from data_sources import small_image_data_source
from architectures.model import Model
from tensorflow.python.tools import freeze_graph
from architectures import regularizers

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset', default='cifar10',
    help=('Dataset to use (cifar10|cifar100|svhn)'))

flags.DEFINE_integer(
    'batch_size', default=256,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=1,
    help=('Number of training epochs.'))

flags.DEFINE_float(
    'learning_rate', default=0.05,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'sgd_momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_bool(
    'sgd_nesterov', default=True,
    help=('Use Nesterov momentum.'))

flags.DEFINE_string(
    'lr_sched_steps', default='[[120, 0.2], [240, 0.04]]',
    help=('Learning rate schedule steps as a Python list; '
          '[[step1_epoch, step1_lr_scale], '
          '[step2_epoch, step2_lr_scale], ...]'))

flags.DEFINE_float(
    'l2_reg', default=0.0005,
    help=('The amount of L2-regularization to apply.'))

flags.DEFINE_integer(
    'n_val', default=0,
    help=('Number of samples to split off the training set for validation.'))

flags.DEFINE_integer(
    'n_sup', default=1000,
    help=('Number of samples to be used for supervised loss (-1 for all).'))

flags.DEFINE_float(
    'teacher_alpha', default=0.97,
    help=('Teacher EMA alpha.'))

flags.DEFINE_string(
    'unsup_reg', default='none',
    help=('Unsupervised/perturbation regularizer '
          '(none|cowout).'))

flags.DEFINE_float(
    'cons_weight', default=1.0,
    help=('Consistency (perturbation) loss weight.'))

flags.DEFINE_float(
    'conf_thresh', default=0.97,
    help=('Consistency (perturbation) confidence threshold.'))

flags.DEFINE_bool(
    'conf_avg', default=False,
    help=('Consistency (perturbation) confidence mask averaging.'))

flags.DEFINE_float(
    'cut_backg_noise', default=1.0,
    help=('Consistency (perturbation) cut background noise (e.g. 1.0 for '
          'RandErase).'))

flags.DEFINE_float(
    'cut_prob', default=1.0,
    help=('Consistency (perturbation) cut probability.'))

flags.DEFINE_string(
    'cow_sigma_range', default='4.0:16.0',
    help=('Consistency (perturbation), unsup_reg is cowout/aug_coowout: the '
          'range of the Gaussian smoothing sigma that controls the scale of '
          'CowMask'))

flags.DEFINE_string(
    'cow_prop_range', default='0.25:1.0',
    help=('Consistency (perturbation), unsup_reg is cowout/aug_coowout: the '
          'range of proportion of the image to be masked out by CowMask'))

flags.DEFINE_string(
    'mix_reg', default='cowmix',
    help=('Mix regularizer '
          '(none|ict|cutmix|cowmix).'))

flags.DEFINE_bool(
    'mix_aug_separately', default=False,
    help=('Mix regularization, use different augmentations for teacher '
          '(unmixed) and student (mixed) paths'))

flags.DEFINE_bool(
    'mix_logits', default=False,
    help=('Mix regularization, mix pre-softmax logits rather than '
          'post-softmax probabilities'))

flags.DEFINE_float(
    'mix_weight', default=30.0,
    help=('Mix regularization, mix consistency loss weight.'))

flags.DEFINE_float(
    'mix_conf_thresh', default=0.6,
    help=('Mix regularization, confidence threshold.'))

flags.DEFINE_bool(
    'mix_conf_avg', default=True,
    help=('Mix regularization, average confidence threshold masks'))

flags.DEFINE_string(
    'mix_conf_mode', default='mix_conf',
    help=('Mix either confidence or probabilities for confidence '
          'thresholding (prob|conf).'))

flags.DEFINE_string(
    'mix_cow_sigma_range', default='4.0:16.0',
    help=('Mix regularization, mix_reg=cowmix: the '
          'range of the Gaussian smoothing sigma that controls the scale of '
          'CowMask'))

flags.DEFINE_string(
    'mix_cow_prop_range', default='0.2:0.8',
    help=('Mix regularization, mix_reg=cowmix: the '
          'range of proportion of the image to be masked out by CowMask'))

flags.DEFINE_integer(
    'subset_seed', default=12345,
    help=('Random seed used to choose supervised samples (n_sup != -1).'))

flags.DEFINE_integer(
    'val_seed', default=131,
    help=('Random seed used to choose validation samples (when n_val > 0).'))

flags.DEFINE_string(
    'dataset_path', default='./dataset',
    help=('Directory to store model data'))

flags.DEFINE_bool(
    'ckpt', default=False,
    help=('Checkpoint exists.'))

flags.DEFINE_string(
    'model_path', default='./checkpoint/',
    help=('Directory to store model data'))


#load dataset
def dataset_load(dataset='cifar10',n_val=5000, n_sup=1000, batch_size=256,
            eval_batch_size=256, augment_twice=False, subset_seed=12345, val_seed=131, dataset_path='./dataset'):
    if dataset == 'svhn':
        data_source = small_image_data_source.SVHNDataSource(
            n_val=n_val, n_sup=n_sup, train_batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            augment_twice=augment_twice,subset_seed=subset_seed,val_seed=val_seed,dataset_path=dataset_path)
    elif dataset == 'cifar10':
        data_source = small_image_data_source.CIFAR10DataSource(
            n_val=n_val, n_sup=n_sup, train_batch_size=batch_size,
            eval_batch_size=eval_batch_size, augment_twice=augment_twice,subset_seed=subset_seed,val_seed=val_seed,dataset_path=dataset_path)
    elif dataset == 'cifar100':
        data_source = small_image_data_source.CIFAR100DataSource(
            n_val=n_val, n_sup=n_sup, train_batch_size=batch_size,
            eval_batch_size=eval_batch_size, augment_twice=augment_twice,subset_seed=subset_seed,val_seed=val_seed,dataset_path=dataset_path)
    else:
        raise RuntimeError
    return data_source

#cowout
def build_pert_reg(unsupervised_regularizer ='cowout', cut_backg_noise=1.0,
                   cut_prob=1.0,
                   cow_sigma_range=(4.0, 8.0), cow_prop_range=(0.0, 1.0)):
    if unsupervised_regularizer == 'none':
        unsup_reg = None
        augment_twice = False
    elif unsupervised_regularizer == 'cowout':
        unsup_reg = regularizers.CowMaskRegularizer(
            cut_backg_noise, cut_prob, cow_sigma_range, cow_prop_range)
        augment_twice = False
    else:
        raise ValueError('Unknown supervised_regularizer \'{}\''.format(
            unsupervised_regularizer))
    return unsup_reg, augment_twice
#cowmix
def build_mix_reg(mix_regularizer ='cowmix',
                  cow_sigma_range=(4.0, 8.0), cow_prop_range=(0.0, 1.0)):
    if mix_regularizer == 'none':
        mix_reg = None
    elif mix_regularizer == 'cowmix':
        mix_reg = regularizers.CowMaskRegularizer(
            0.0, 1.0, cow_sigma_range, cow_prop_range)
    else:
        raise ValueError('Unknown supervised_regularizer \'{}\''.format(
            mix_regularizer))
    return mix_reg

def train( dataset='cifar10',
               batch_size=128,
               num_epochs=300,
               learning_rate=0.05,
               sgd_momentum=0.9,
               sgd_nesterov=True,
               lr_sched_steps=[[120, 0.2], [240, 0.04]],
               l2_reg=0.0005,
               n_val=5000,#
               n_sup=0,#
               ema=0.97,#ema计算teacher的参数
               unsup_regularizer='none',
               cons_weight=1.0,
               conf_thresh=0.97,
               conf_avg=False,
               cut_backg_noise=1.0,
               cut_prob=1.0,
               cow_sigma_range=(4.0, 16.0),
               cow_prop_range=(0.25, 1.0),
               mix_regularizer='cowmix',
               mix_aug_separately=False,
               mix_logits=False,
               mix_weight=30.0,
               mix_conf_thresh=0.6,
               mix_conf_avg=True,
               mix_conf_mode='mix_prob',
               mix_cow_sigma_range=(4.0, 16.0),
               mix_cow_prop_range=(0.2, 0.8),
               subset_seed=12345,
               val_seed=131,
               dataset_path='./dataset/',
               ckpt=False,
               model_path=None
               ):

    # Mask-based erasure
    unsup_reg, augment_twice = build_pert_reg(unsupervised_regularizer =unsup_regularizer, cut_backg_noise=cut_backg_noise, cut_prob=cut_prob, cow_sigma_range=cow_sigma_range, cow_prop_range=cow_prop_range)
    # Mask-based mixing
    mix_reg = build_mix_reg(mix_regularizer=mix_regularizer, cow_sigma_range=mix_cow_sigma_range, cow_prop_range=mix_cow_prop_range)

    data_source = dataset_load(dataset=dataset, n_val=n_val, n_sup=n_sup, batch_size=batch_size, eval_batch_size= batch_size, augment_twice=augment_twice,subset_seed=subset_seed,val_seed=val_seed,dataset_path=dataset_path)
    image_size = data_source.image_size
    n_train = data_source.n_train
    train_ds=data_source.train_semisup_ds
    if n_val == 0:
        n_eval = data_source.n_eval
        eval_ds = data_source.test_ds
    else:
        n_eval = data_source.n_eval
        eval_ds = data_source.val_ds
    tf.reset_default_graph()
    logging.info('DATA: |train|={}, |sup|={}, |eval|={}, (|val|={}, |test|={})'.format(
        data_source.n_train, data_source.n_sup, n_eval, data_source.n_val,
        data_source.n_test))
    logging.info('Loaded dataset')

    steps_per_epoch = n_train // batch_size
    steps_per_eval = n_eval // batch_size

    model = Model(batch_size,
                  num_epochs,
                  image_size,
                  data_source.n_classes,
                  train_ds,
                  eval_ds,
                  steps_per_epoch,
                  steps_per_eval,
                  learning_rate,
                  lr_sched_steps,
                  l2_reg,
                  ema,
                  sgd_momentum,
                  sgd_nesterov,
                  unsup_reg=unsup_reg,
                  cons_weight=cons_weight,
                  conf_thresh=conf_thresh,
                  conf_avg=conf_avg,
                  mix_reg=mix_reg,
                  mix_logits=mix_logits,
                  mix_weight=mix_weight,
                  mix_conf_thresh=mix_conf_thresh,
                  mix_conf_avg=mix_conf_avg,
                  mix_conf_mode=mix_conf_mode,
                  mix_aug_separately=mix_aug_separately,
                  ckpt=ckpt,
                  model_path=model_path
                  )
    logits = model.operas['val_logits_tea']
    out_put = tf.argmax(logits, axis=1, output_type=tf.int32, name="output") #output node will be used to inference
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'output_empty.pb')  # save pb file with output node
        freeze_graph.freeze_graph(
            input_graph='./pb_model/output_empty.pb',  # the pb file with output node
            input_saver='',
            input_binary=False,
            input_checkpoint=model_path+'milking_cowmask.ckpt-300',  # input checkpoint file path
            output_node_names='output',  # the name of output node in pb file
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/milking_cowmask.pb',  # path of output graph
            clear_devices=False,
            initializer_nodes='')
    logging.info('done')

def _range_str_to_tuple(s):
  xs = [x.strip() for x in s.split(':')]
  return tuple([float(x) for x in xs])

def main(argv):
    if len(argv) > 34:
        raise app.UsageError('Too many command-line arguments.')
    if not tf.gfile.Exists(FLAGS.model_path):
        tf.gfile.MkDir(FLAGS.model_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    train(
        dataset=FLAGS.dataset,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        sgd_momentum=FLAGS.sgd_momentum,
        sgd_nesterov=FLAGS.sgd_nesterov,
        lr_sched_steps=ast.literal_eval(FLAGS.lr_sched_steps),
        l2_reg=FLAGS.l2_reg,
        n_val=FLAGS.n_val,
        n_sup=FLAGS.n_sup,
        ema=FLAGS.teacher_alpha,  # ema计算teacher的参数
        unsup_regularizer=FLAGS.unsup_reg,
        cons_weight=FLAGS.cons_weight,
        conf_thresh=FLAGS.conf_thresh,
        conf_avg=FLAGS.conf_avg,
        cut_backg_noise=FLAGS.cut_backg_noise,
        cut_prob=FLAGS.cut_prob,
        cow_sigma_range=_range_str_to_tuple(FLAGS.cow_sigma_range),
        cow_prop_range=_range_str_to_tuple(FLAGS.cow_prop_range),
        mix_regularizer=FLAGS.mix_reg,
        mix_aug_separately=FLAGS.mix_aug_separately,
        mix_logits=FLAGS.mix_logits,
        mix_weight=FLAGS.mix_weight,
        mix_conf_thresh=FLAGS.mix_conf_thresh,
        mix_conf_avg=FLAGS.mix_conf_avg,
        mix_conf_mode=FLAGS.mix_conf_mode,
        subset_seed=FLAGS.subset_seed,
        val_seed=FLAGS.val_seed,
        dataset_path=FLAGS.dataset_path,
        ckpt=FLAGS.ckpt,
        model_path=FLAGS.model_path
    )

if __name__ == '__main__':
    app.run(main)

