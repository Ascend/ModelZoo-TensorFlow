import os,sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Input, Conv2D, ReLU, LeakyReLU
from modules.anchor import decode_tf, prior_box_tf


def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 name=None,
                 trainable=True,
                 **kwargs):
        super(BatchNormalization, self).__init__(axis=axis,
                                                 momentum=momentum,
                                                 epsilon=epsilon,
                                                 center=center,
                                                 scale=scale,
                                                 name=name,
                                                 trainable=True,
                                                 **kwargs)

    def call(self, x, training=False):
        # def call(self, x, training=True):
        if training is None:
            training = tf.constant(False)
            # training = tf.constant(True)
        training = tf.logical_and(training, self.trainable)

        return super().call(x, training)




def Backbone(backbone_type='ResNet50', use_pretrain=True, use_local_pretrain=True, root_path=''):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        if not use_local_pretrain:
            weights = 'imagenet'
        else:
            mbnv2_notop_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
            res50_notop_name = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
            models_default_floder = os.path.join(root_path, 'pretrain_models')
            weights = os.path.join(models_default_floder, res50_notop_name) if backbone_type == 'ResNet50' else os.path.join(models_default_floder, mbnv2_notop_name)


    def backbone(x):
        if backbone_type == 'ResNet50':
            shape_input = x.shape.as_list()[1:] if weights != 'imagenet' else x.shape[1:]
            extractor = ResNet50(
                input_shape=shape_input, include_top=False, weights=weights
            )
            pick_layer1 = 80  # [80, 80, 512]
            pick_layer2 = 142  # [40, 40, 1024]
            pick_layer3 = 174  # [20, 20, 2048]
            preprocess = tf.keras.applications.resnet.preprocess_input
        elif backbone_type == 'MobileNetV2':
            shape_input = x.shape.as_list()[1:] if weights != 'imagenet' else x.shape[1:]
            extractor = MobileNetV2(
                input_shape=shape_input, include_top=False, weights=weights)
            pick_layer1 = 54  # [80, 80, 32]
            pick_layer2 = 116  # [40, 40, 96]
            pick_layer3 = 143  # [20, 20, 160]
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        else:
            raise NotImplementedError(
                'Backbone type {} is not recognized.'.format(backbone_type))

        return Model(
            extractor.input,
                    (
                        extractor.layers[pick_layer1].output,
                        extractor.layers[pick_layer2].output,
                        extractor.layers[pick_layer3].output
                    ),
                     name=backbone_type + '_extrator')(preprocess(x))

    return backbone


class ConvUnit(tf.keras.layers.Layer):
    """Conv + BN + Act"""
    def __init__(self, f, k, s, wd, act=None, training=False, name='ConvBN', **kwargs):
        super(ConvUnit, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=f, kernel_size=k, strides=s, padding='same',
                           kernel_initializer=_kernel_init(),
                           kernel_regularizer=_regularizer(wd),
                           use_bias=False, name='conv')
        self.training = training

        if act is None:
            self.act_fn = tf.identity
        elif act == 'relu':
            self.act_fn = ReLU()
        elif act == 'lrelu':
            self.act_fn = LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))
        #     # original bn

        # def call(self, x):
        #     return self.act_fn(tf.layers.batch_normalization(self.conv(x),
        #                                         training=self.training,
        #                                         epsilon=1e-5))

        # keras bn
        self.bn = BatchNormalization(name='bn')

    def call(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(tf.keras.layers.Layer):
    """Feature Pyramid Network"""
    def __init__(self, out_ch, wd, training=False, name='FPN', **kwargs):
        super(FPN, self).__init__(name=name, **kwargs)
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.output1 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, training=training)
        self.output2 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, training=training)
        self.output3 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, training=training)
        self.merge1 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act, training=training)
        self.merge2 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act, training=training)

    def call(self, x):
        output1 = self.output1(x[0])  # [80, 80, out_ch]
        output2 = self.output2(x[1])  # [40, 40, out_ch]
        output3 = self.output3(x[2])  # [20, 20, out_ch]

        up_h, up_w = tf.shape(output2)[1], tf.shape(output2)[2]
        up3 = tf.image.resize(output3, [up_h, up_w], method='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tf.shape(output1)[1], tf.shape(output1)[2]
        up2 = tf.image.resize(output2, [up_h, up_w], method='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3


class SSH(tf.keras.layers.Layer):
    """Single Stage Headless Layer"""
    def __init__(self, out_ch, wd, training=False, name='SSH', **kwargs):
        super(SSH, self).__init__(name=name, **kwargs)
        assert out_ch % 4 == 0
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.conv_3x3 = ConvUnit(f=out_ch // 2, k=3, s=1, wd=wd, act=None, training=training)

        self.conv_5x5_1 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act, training=training)
        self.conv_5x5_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None, training=training)

        self.conv_7x7_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act, training=training)
        self.conv_7x7_3 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None, training=training)

        self.relu = ReLU()

    def call(self, x):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tf.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = self.relu(output)

        return output


class BboxHead(tf.keras.layers.Layer):
    """Bbox Head Layer"""
    def __init__(self, num_anchor, wd, name='BboxHead', **kwargs):
        super(BboxHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 4, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 4])


class LandmarkHead(tf.keras.layers.Layer):
    """Landmark Head Layer"""
    def __init__(self, num_anchor, wd, name='LandmarkHead', **kwargs):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 10, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 10])


class ClassHead(tf.keras.layers.Layer):
    """Class Head Layer"""
    def __init__(self, num_anchor, wd, name='ClassHead', **kwargs):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = Conv2D(filters=num_anchor * 2, kernel_size=1, strides=1)

    def call(self, x):
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = self.conv(x)

        return tf.reshape(x, [-1, h * w * self.num_anchor, 2])


def RetinaFaceModelFunc(cfg, training=False, iou_th=0.4, score_th=0.02,
                    name='RetinaFaceModel'):
    """Retina Face Model"""
    input_size = cfg['input_size'] if training else None
    wd = cfg['weights_decay']
    out_ch = cfg['out_channel']
    num_anchor = len(cfg['min_sizes'][0])
    backbone_type = cfg['backbone_type']
    root_path = cfg['root_path']

    # define model
    x = inputs = Input([input_size, input_size, 3], name='input_image')
    # x = inputs = input_holder

    x = Backbone(backbone_type=backbone_type, root_path=root_path)(x)

    fpn = FPN(out_ch=out_ch, wd=wd, training=training)(x)

    features = [SSH(out_ch=out_ch, wd=wd, training=training, name=f'SSH_{i}')(f)
                for i, f in enumerate(fpn)]

    bbox_regressions = tf.concat(
        [BboxHead(num_anchor, wd=wd, name=f'BboxHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)
    landm_regressions = tf.concat(
        [LandmarkHead(num_anchor, wd=wd, name=f'LandmarkHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)
    classifications = tf.concat(
        [ClassHead(num_anchor, wd=wd, name=f'ClassHead_{i}')(f)
         for i, f in enumerate(features)], axis=1)

    classifications = tf.keras.layers.Softmax(axis=-1)(classifications)

    if training:
        out = (bbox_regressions, landm_regressions, classifications)
    else:
        # only for batch size 1
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [bbox_regressions[0], landm_regressions[0],
             tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
             classifications[0, :, 1][..., tf.newaxis]], 1)
        priors = prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
                              cfg['min_sizes'],  cfg['steps'], cfg['clip'])
        decode_preds = decode_tf(preds, priors, cfg['variances'])

        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=iou_th,
            score_threshold=score_th)

        out = tf.gather(decode_preds, selected_indices)

    return Model(inputs, out, name=name)

class RetinaFaceModel(Model):
    def __init__(self, cfg, training=False, iou_th=0.4, score_th=0.02, name='RetinaFaceModel'):
        super().__init__(name=name)
        self.training = training
        # test cfg
        self.score_th = score_th
        self.iou_th = iou_th
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.variances = cfg['variances']
        # training cfg
        self.wd = cfg['weights_decay']
        self.out_ch = cfg['out_channel']
        self.num_anchor = len(cfg['min_sizes'][0])
        self.backbone_type = cfg['backbone_type']
        self.root_path = cfg['root_path']
        # build network layer
        self.backbone = Backbone(backbone_type=self.backbone_type, root_path=self.root_path)
        self.fpn = FPN(out_ch=self.out_ch, wd=self.wd, training=self.training)
        self.sshs = [SSH(out_ch=self.out_ch, wd=self.wd, training=self.training, name=f'SSH_{i}') for i in range(3)]
        self.bboxheads = [BboxHead(self.num_anchor, wd=self.wd, name=f'BboxHead_{i}') for i in range(3)]
        self.landmarkheads = [LandmarkHead(self.num_anchor, wd=self.wd, name=f'LandmarkHead_{i}') for i in range(3)]
        self.classheads = [ClassHead(self.num_anchor, wd=self.wd, name=f'ClassHead_{i}') for i in range(3)]
        self.classifications = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        # build graph
        x = self.backbone(inputs)
        fpn = self.fpn(x)
        features = [self.sshs[i](f) for i, f in enumerate(fpn)]
        bbox_regressions = tf.concat([self.bboxheads[i](f) for i, f in enumerate(features)], axis=1)
        landm_regressions = tf.concat([self.landmarkheads[i](f) for i, f in enumerate(features)], axis=1)
        classifications = tf.concat([self.classheads[i](f) for i, f in enumerate(features)], axis=1)
        classifications = self.classifications(classifications)

        if self.training:
            out = (bbox_regressions, landm_regressions, classifications)
        else:
            # only for batch size 1
            # concat [bboxes, landms, landms_valid, conf]
            preds = tf.concat([bbox_regressions[0], landm_regressions[0],
                tf.ones_like(classifications[0, :, 0][..., tf.newaxis]),
                classifications[0, :, 1][..., tf.newaxis]], 1)
            priors = prior_box_tf((tf.shape(inputs)[1], tf.shape(inputs)[2]),
                                self.min_sizes, self.steps, self.clip)
            decode_preds = decode_tf(preds, priors, self.variances)

            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=self.iou_th,
                score_threshold=self.score_th)

            out = tf.gather(decode_preds, selected_indices)
        return out
