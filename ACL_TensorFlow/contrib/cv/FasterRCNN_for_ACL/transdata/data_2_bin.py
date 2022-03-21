# Copyright 2020 Huawei Technologies Co., Ltd
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

from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import sys,os
import six

sys.path.insert(0, 'tpu/models')

import dataloader_infer as dataloader
import mask_rcnn_model
import spatial_transform_ops
import preprocess_ops
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from object_detection import tf_example_decoder
import anchors
import coco_utils
import preprocess_ops
import spatial_transform_ops

#from configs import mask_rcnn_config
#from configs import mask_rcnn_config_8gpus_1333x800 as mask_rcnn_config
#from configs import mask_rcnn_config_1gpu as mask_rcnn_config
#from configs import mask_rcnn_config_8gpus_1024x1024 as mask_rcnn_config
from configs import fast_rcnn_config_8gpus_1024x1024 as mask_rcnn_config
from mask_rcnn_model import build_model_graph
from mask_rcnn_model import _model_fn
common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

MAX_NUM_INSTANCES = 100
MAX_NUM_VERTICES_PER_INSTANCE = 1500
MAX_NUM_POLYGON_LIST_LEN = 2 * MAX_NUM_VERTICES_PER_INSTANCE * MAX_NUM_INSTANCES
max_num_instances = MAX_NUM_INSTANCES
max_num_polygon_list_len = MAX_NUM_POLYGON_LIST_LEN

flags.DEFINE_string(
    'distribution_strategy',
    default='multi_worker_gpu',
    help='Distribution strategy or estimator type to use. One of'
    '"multi_worker_gpu"|"tpu".')

# Parameters for MultiWorkerMirroredStrategy
flags.DEFINE_string(
    'worker_hosts',
    default=None,
    help='Comma-separated list of worker ip:port pairs for running '
    'multi-worker models with distribution strategy.  The user would '
    'start the program on each host with identical value for this flag.')
flags.DEFINE_integer(
    'task_index', 0, 'If multi-worker training, the task_index of this worker.')
flags.DEFINE_integer(
    'num_gpus',
    default=1,
    help='Number of gpus when using collective all reduce strategy.')
flags.DEFINE_integer(
    'worker_replicas',
    default=0,
    help='Number of workers when using collective all reduce strategy.')

# TPUEstimator parameters
flags.DEFINE_integer(
    'num_cores', default=None, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_bool(
    'transpose_input',
    default=None,
    help='Use TPU double transpose optimization')
flags.DEFINE_string(
    'tpu_job_name', None,
    'Name of TPU worker binary. Only necessary if job name is changed from'
    ' default tpu_worker.')

# Model specific paramenters
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_string(
    'training_file_pattern', '/root/33334/data_reMake_includeMask/train-000*',
    ' ')

flags.DEFINE_string(
    'resnet_checkpoint', '/root/33334/resnet34/model.ckpt-28152',
    ' ')

flags.DEFINE_string(
    'validation_file_pattern', '/home/zl/fast-rcnn/data_new/coco_official_2017/tfrecord/val*',
    ' ')

# modelarts
flags.DEFINE_string(
    'data_url', None,
    'path to dataset.')
flags.DEFINE_string(
    'train_url', None,
    'train_dir')


FLAGS = flags.FLAGS
FLAGS(sys.argv)
# def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, idx):
#     token = batch_tokens[idx]
#     predict = id2label[prediction]
#     true_l = id2label[batch_labels[idx]]
#     if token != "[PAD]" and token != "[CLS]" and true_l != "X":
#         if predict == "X" and not predict.startswith("##"):
#             predict = "O"
#         line = "{}\t{}\t{}\n".format(token, true_l, predict)
#         wf.write(line)

# def read_and_decode(filename):  # 读入tfrecords
#     filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })  # 将image数据和label取出�?#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [1024, 1024, 3])
#     # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
#     label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
#     print('img',img)
#     print('label', label)
#     return img, label
#
# def create_example_decoder(use_instance_mask):
#     return tf_example_decoder.TfExampleDecoder(
#         use_instance_mask=use_instance_mask)
#
# def create_dataset_parser_fn(params,mode):
#     use_instance_mask = params.include_mask
#     """Create parser for parsing input data (dictionary)."""
#     example_decoder = create_example_decoder(use_instance_mask)
#
#     with tf.name_scope('parser'):
#         data = example_decoder.decode(value)
#         data['groundtruth_is_crowd'] = tf.cond(
#             tf.greater(tf.size(data['groundtruth_is_crowd']), 0),
#             lambda: data['groundtruth_is_crowd'],
#             lambda: tf.zeros_like(data['groundtruth_classes'], dtype=tf.bool))
#         filename = '/home/zhouli/fast-rcnn/data_new/coco_official_2017/tfrecord/val2017_tfrecord-00000-of-00032.tfrecord'
#         # image = data['image']
#         image = read_and_decode(filename)
#         image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         orig_image = image
#         source_id = data['source_id']
#         source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
#                              source_id)
#         source_id = tf.string_to_number(source_id)
#
#         if (mode == tf.estimator.ModeKeys.PREDICT or
#                     mode == tf.estimator.ModeKeys.EVAL):
#             image = preprocess_ops.normalize_image(image)
#             if params['resize_method'] == 'retinanet':
#                 image, image_info, _, _, _ = preprocess_ops.resize_crop_pad(
#                     image, params['image_size'], 2 ** params['max_level'])
#             else:
#                 image, image_info, _, _, _ = preprocess_ops.resize_crop_pad_v2(
#                     image, params['short_side'], params['long_side'],
#                     2 ** params['max_level'])
#             if params['precision'] == 'bfloat16':
#                 image = tf.cast(image, dtype=tf.bfloat16)
#
#             features = {
#                 'images': image,
#                 'image_info': image_info,
#                 'source_ids': source_id,
#             }
#             if params['visualize_images_summary']:
#                 resized_image = tf.image.resize_images(orig_image,
#                                                        params['image_size'])
#                 features['orig_images'] = resized_image
#             if (params['include_groundtruth_in_features'] or
#                         mode == tf.estimator.ModeKeys.EVAL):
#                 labels = _prepare_labels_for_eval(
#                     data,
#                     target_num_instances=max_num_instances,
#                     target_polygon_list_len=max_num_polygon_list_len,
#                     use_instance_mask=params['include_mask'])
#                 return {'features': features, 'labels': labels}
#             else:
#                 return {'features': features}
#
#         elif mode == tf.estimator.ModeKeys.TRAIN:
#             instance_masks = None
#             if use_instance_mask:
#                 instance_masks = data['groundtruth_instance_masks']
#             boxes = data['groundtruth_boxes']
#             classes = data['groundtruth_classes']
#             classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
#             if not params['use_category']:
#                 classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
#
#             if (params['skip_crowd_during_training'] and
#                         mode == tf.estimator.ModeKeys.TRAIN):
#                 indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
#                 classes = tf.gather_nd(classes, indices)
#                 boxes = tf.gather_nd(boxes, indices)
#                 if use_instance_mask:
#                     instance_masks = tf.gather_nd(instance_masks, indices)
#
#             image = preprocess_ops.normalize_image(image)
#             if params['input_rand_hflip']:
#                 flipped_results = (
#                     preprocess_ops.random_horizontal_flip(
#                         image, boxes=boxes, masks=instance_masks))
#                 if use_instance_mask:
#                     image, boxes, instance_masks = flipped_results
#                 else:
#                     image, boxes = flipped_results
#             # Scaling, jittering and padding.
#             if params['resize_method'] == 'retinanet':
#                 image, image_info, boxes, classes, cropped_gt_masks = (
#                     preprocess_ops.resize_crop_pad(
#                         image,
#                         params['image_size'],
#                         2 ** params['max_level'],
#                         aug_scale_min=params['aug_scale_min'],
#                         aug_scale_max=params['aug_scale_max'],
#                         boxes=boxes,
#                         classes=classes,
#                         masks=instance_masks,
#                         crop_mask_size=params['gt_mask_size']))
#             else:
#                 image, image_info, boxes, classes, cropped_gt_masks = (
#                     preprocess_ops.resize_crop_pad_v2(
#                         image,
#                         params['short_side'],
#                         params['long_side'],
#                         2 ** params['max_level'],
#                         aug_scale_min=params['aug_scale_min'],
#                         aug_scale_max=params['aug_scale_max'],
#                         boxes=boxes,
#                         classes=classes,
#                         masks=instance_masks,
#                         crop_mask_size=params['gt_mask_size']))
#             if cropped_gt_masks is not None:
#                 cropped_gt_masks = tf.pad(
#                     cropped_gt_masks,
#                     paddings=tf.constant([[0, 0, ], [2, 2, ], [2, 2]]),
#                     mode='CONSTANT',
#                     constant_values=0.)
#
#             padded_height, padded_width, _ = image.get_shape().as_list()
#             padded_image_size = (padded_height, padded_width)
#             input_anchors = anchors.Anchors(
#                 params['min_level'],
#                 params['max_level'],
#                 params['num_scales'],
#                 params['aspect_ratios'],
#                 params['anchor_scale'],
#                 padded_image_size)
#             anchor_labeler = anchors.AnchorLabeler(
#                 input_anchors,
#                 params['num_classes'],
#                 params['rpn_positive_overlap'],
#                 params['rpn_negative_overlap'],
#                 params['rpn_batch_size_per_im'],
#                 params['rpn_fg_fraction'])
#
#             # Assign anchors.
#             score_targets, box_targets = anchor_labeler.label_anchors(
#                 boxes, classes)
#
#             # Pad groundtruth data.
#             boxes = preprocess_ops.pad_to_fixed_size(
#                 boxes, -1, [max_num_instances, 4])
#             classes = preprocess_ops.pad_to_fixed_size(
#                 classes, -1, [max_num_instances, 1])
#
#             # Pads cropped_gt_masks.
#             if use_instance_mask:
#                 cropped_gt_masks = tf.reshape(
#                     cropped_gt_masks, tf.stack([tf.shape(cropped_gt_masks)[0], -1]))
#                 cropped_gt_masks = preprocess_ops.pad_to_fixed_size(
#                     cropped_gt_masks, -1,
#                     [max_num_instances, (params['gt_mask_size'] + 4) ** 2])
#                 cropped_gt_masks = tf.reshape(
#                     cropped_gt_masks,
#                     [max_num_instances, params['gt_mask_size'] + 4,
#                      params['gt_mask_size'] + 4])
#
#             if params['precision'] == 'bfloat16':
#                 image = tf.cast(image, dtype=tf.bfloat16)
#
#             features = {
#                 'images': image,
#                 'image_info': image_info,
#                 'source_ids': source_id,
#             }
#
#             labels = {}
#             for level in range(params['min_level'], params['max_level'] + 1):
#                 labels['score_targets_%d' % level] = score_targets[level]
#                 labels['box_targets_%d' % level] = box_targets[level]
#             labels['gt_boxes'] = boxes
#             labels['gt_classes'] = classes
#             if use_instance_mask:
#                 labels['cropped_gt_masks'] = cropped_gt_masks
#             print('wwwwwwwwww', features)
#             return features, labels

def do_bm_predict():
    # predict_list = []
    # for i in range(len(ids_list)):
    #     tf.compat.v1.reset_default_graph()
    #     config = tf.ConfigProto()
    params = params_dict.ParamsDict(
        mask_rcnn_config.MASK_RCNN_CFG, mask_rcnn_config.MASK_RCNN_RESTRICTIONS)
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
    params = params_dict.override_params_dict(
        params, FLAGS.params_override, is_strict=True)
    params = flags_to_params.override_params_from_input_flags(params, FLAGS)
    params.validate()
    params.lock()

    print("qqqqqqqqqq", params)
    print("qqqqqqqqqq", params.validation_file_pattern)

    # params = dict(params.as_dict().items(),
    #               input_rand_hflip=False,
    #               is_training_bn=False,
    #               transpose_input=False)
    # print("22222", params)
    # with tf.gfile.FastGFile(os.path.join(FLAGS.pb_model_file), "rb") as f:
    #     graph_def = tf.compat.v1.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #     sess.graph.as_default()
    #     tf.import_graph_def(graph_def, name='')

    # print("input ids file name: %s" % ids_list[i])
    # input_ids = tf.compat.v1.get_default_graph().get_tensor_by_name("image:0")
    # input_mask = tf.compat.v1.get_default_graph().get_tensor_by_name("image_info:0")
    # input_segment = tf.compat.v1.get_default_graph().get_tensor_by_name("source_id:0")
    graph = tf.Graph()
    with graph.as_default():
        newparam = {}
        newparam['validation_file_pattern'] = params.validation_file_pattern
        newparam['resize_method'] = params.resize_method
        newparam['image_size'] =  params.image_size
        newparam['max_level'] = params.max_level
        newparam['short_side'] = params.short_side
        newparam['long_side'] =params.long_side
        newparam['max_level'] =params.max_level
        newparam['precision'] =params.precision
        newparam['include_groundtruth_in_features'] =params.include_groundtruth_in_features
        newparam['visualize_images_summary']= params.visualize_images_summary
        newparam['include_mask']= params.include_mask
        newparam['use_category'] = params.use_category
        newparam['skip_crowd_during_training'] = params.skip_crowd_during_training
        newparam['input_rand_hflip'] = params.input_rand_hflip
        newparam['resize_method'] = params.resize_method
        newparam['aug_scale_min'] = params.aug_scale_min
        newparam['aug_scale_max'] = params.aug_scale_max
        newparam['gt_mask_size'] = params.gt_mask_size
        newparam['short_side'] = params.short_side
        newparam['long_side'] = params.long_side
        newparam['num_scales'] = params.num_scales
        newparam['aspect_ratios'] = params.aspect_ratios
        newparam['anchor_scale'] = params.anchor_scale
        newparam['num_classes'] = params.num_classes
        newparam['rpn_positive_overlap'] = params.rpn_positive_overlap
        newparam['rpn_negative_overlap'] = params.rpn_negative_overlap
        newparam['rpn_batch_size_per_im'] = params.rpn_batch_size_per_im
        newparam['rpn_fg_fraction'] = params.rpn_fg_fraction
        newparam['precision'] = params.precision
        newparam['transpose_input'] = params.transpose_input
        newparam['backbone'] = params.backbone
        newparam['conv0_space_to_depth_block_size'] = params.conv0_space_to_depth_block_size
        # newparam['num_classes'] = params.num_classes
        # newparam['num_classes'] = params.num_classes
        # newparam['num_classes'] = params.num_classes
        eval_input_fn = dataloader.InputReader(
            params.validation_file_pattern,
            mode=tf.estimator.ModeKeys.PREDICT,
            num_examples=params.eval_samples,
            use_instance_mask=params.include_mask)
        dataset = eval_input_fn(newparam)
        # dataset = dataset.take(1)
        # print('...........', dataset.take(1))
        #
        iterator = dataset.make_initializable_iterator()
        #
        # a = 2
        # while (a > 0):
        #     try:
        features = iterator.get_next()
        print('...........', features)
        print('...........', features['features'])
        print('...........', features['features']['images'])
        print('...........', features['features']['image_info'])
        print('...........', features['features']['source_ids'])
        images = features['features']['images']
        image_info = features['features']['image_info']
        source_ids = features['features']['source_ids']
        images = tf.cast(images, tf.float32)
        image_info = tf.cast(image_info, tf.float32)
        source_ids = tf.cast(source_ids, tf.float32)
                # a = a - 1
            #     print('~~~~a',a)
            # except:
            #     print('iteraotr is null~~~~~')
            #     print('~~~~a', a)
            #     break
        # out1 = tf.nn.relu(images)
        # out2 = tf.nn.relu(image_info)

        # print('...........', )
        # while True:
        #     try:
        #         prediction = six.next(dataset)
        #         out = prediction
        #     except StopIteration:
        #         logging.info('Finished the eval set at %d batch.', (batch_idx + 1))
        #         break
    with tf.Session(graph=graph) as sess:
        sess.run(iterator.initializer)
        ouput = sess.run([images, image_info, source_ids])
        print('...........', ouput)
        a = 0
        while (a < 5000):
            try:
                ouput = sess.run([images,image_info,source_ids])
                print(a,'1111111',ouput[0])
                print(a,'2222222', ouput[1])
                print(a,'3333333', ouput[2])
                ouput[0].tofile(os.path.join("./data", "{}_images.bin".format(a)))
                ouput[1].tofile(os.path.join("./data", "{}_image_info.bin".format(a)))
                ouput[2].tofile(os.path.join("./data", "{}_source_ids.bin".format(a)))
                print('...........',ouput)
                a = a + 1
                print('~~~~a',a)
            except:
                print('iteraotr is null~~~~~')
                print('~~~~a', a)
                break
        # feature = eval_input_fn[0]
        # labels = eval_input_fn[1]
        # predict = sess.run(
        #     output,
        #     feed_dict={input_ids: feed_ids, input_mask: feed_mask, input_segment: feed_segment},
        #     options=run_options,
        #     run_metadata=run_metadata)
        # predict_list.append(predict)
        # print("~~~~~~~~",feature)


    # return predict_list

if __name__ == '__main__':
    do_bm_predict()

    # read_and_decode(filename)

