import tensorflow as tf
import os
import pb_net
import argparse
# from utility import custom_op
from tensorflow.python.framework import graph_util

_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def parse_args():
    parser = argparse.ArgumentParser(description="Tensorflow implementation of PyramidBox")

    parser.add_argument('--batch_size', type=int, default=1, help='The batch size')
    parser.add_argument('--img_size', type=int, default=640, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--result', type=str, default='./checkpoint_dir/results', help='Directory name to save the results')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoint_dir", help='Directory name to save the checkpoints')
    parser.add_argument('--pb_dir', type=str, default="./checkpoint_dir/pb")

    return parser.parse_args()


def _ImageDimensions(image, rank = 3):
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]

def _mean_image_subtraction(image, means):
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def preprocess(image, out_shape):
    with tf.name_scope('pyramid_box_preprocessing_eval', 'pyramid_box_preprocessing_eval', [image]):
        image = tf.to_float(image)
        image = tf.image.resize_images(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        image.set_shape(out_shape + [3])

        height, width, _ = _ImageDimensions(image, rank=3)
        output_shape = tf.stack([height, width])

        image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image_channels = tf.unstack(image, axis=-1, name='split_rgb')
        image = tf.stack([image_channels[2], image_channels[1], image_channels[0]], axis=-1, name='merge_bgr')
        # Image data format.
        image = tf.transpose(image, perm=(2, 0, 1))
    return image, output_shape


def freeze_graph(args):
    tf.reset_default_graph()
    out_shape = [args.img_size, args.img_size]
    input = tf.placeholder(tf.uint8, [args.img_size, args.img_size, args.img_ch], name='input')
    input, output_shape = preprocess(input, out_shape)
    input = tf.expand_dims(input, axis=0)

    with tf.Graph().as_default():
        with tf.variable_scope("pyramid_box", default_name = None, values=[input], reuse=tf.AUTO_REUSE):
            backbone = pb_net.VGG16Backbone('channels_first')
            feature_layers = backbone.get_featmaps(input, training=False)
            feature_layers = backbone.build_lfpn(feature_layers, skip_last=3)
            feature_layers = backbone.context_pred_module(feature_layers)
            location_pred, cls_pred, head_location_pred, head_cls_pred, body_location_pred, body_cls_pred = \
                backbone.get_predict_module(feature_layers, name='predict_face')

        cls_pred = tf.reshape(cls_pred, [-1, 2])
        location_pred = tf.reshape(location_pred, [-1, 4])  # 结果1
        cls_pred = tf.nn.softmax(cls_pred)[:, -1]  # 结果2
        tf.identity(location_pred, name='output1')
        tf.identity(cls_pred, name='output2')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver.restore(sess, os.path.join(args.checkpoint_dir, "model.ckpt-290000"))  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['output1', 'output2'])  # 如果有多个输出节点，以逗号隔开
        # 保存模型
        with tf.gfile.GFile(os.path.join(args.pb_dir, "pyramidbox.pb"), "wb") as f:
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    print("done")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # parse arguments
    args = parse_args()

    freeze_graph(args)