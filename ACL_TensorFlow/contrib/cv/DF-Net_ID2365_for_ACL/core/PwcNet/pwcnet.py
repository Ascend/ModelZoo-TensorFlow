# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 20:14:30 2021

@author: B0ba11
"""

from __future__ import absolute_import, division, print_function
import time
import datetime
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange
from tensorflow.contrib.mixed_precision import LossScaleOptimizer, FixedLossScaleManager

from .model_base_gpu import ModelBase
from .optflow import flow_write, flow_write_as_png, flow_mag_stats
from .losses import pwcnet_loss
from .logger import OptFlowTBLogger
from .multi_gpus import assign_to_device, average_gradients
from .core_warp import dense_image_warp
from .core_costvol import cost_volume
from .utils import tf_where


pyr_lvls = 6
dbg = False
search_range = 4
flow_pred_lvl = 2
use_dense_cx = True 
use_res_cx = True
use_mixed_precision = False

def adapt_x(x):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the actual data, not the x TF tensor.
    Args:
        x: input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    assert (isinstance(x, np.ndarray) or isinstance(x, list))
    if isinstance(x, np.ndarray):
        assert (len(x.shape) == 5)
        assert (x.shape[1] == 2 and x.shape[4] == 3)
    else:
        assert (len(x[0].shape) == 4)
        assert (x[0].shape[0] == 2 or x[0].shape[3] == 3)

    # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
    if use_mixed_precision is True:
        x_adapt = np.array(x, dtype=np.float16) if isinstance(x, list) else x.astype(np.float16)
    else:
        x_adapt = np.array(x, dtype=np.float32) if isinstance(x, list) else x.astype(np.float32)
    x_adapt /= 255.

    # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(x_adapt.shape[2], 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(x_adapt.shape[3], 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    x_adapt_info = None
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)]
        x_adapt_info = x_adapt.shape  # Save original shape
        x_adapt = np.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, x_adapt_info
    
def adapt_x_tnsr(x):
    """Preprocess the input samples to adapt them to the network's requirements
    Here, x, is the x TF tensor.
    Args:
        x: (B, 2, 320, 1152, 3)input samples in list[(2,H,W,3)] or (N,2,H,W,3) np array form
    Returns:
        Samples ready to be given to the network (w. same shape as x)
        Also, return adaptation info in (N,2,H,W,3) format
    """
    # Ensure we're dealing with RGB image pairs
    assert (isinstance(x, tf.Tensor))
    if isinstance(x, tf.Tensor):
        assert (len(x.shape) == 5)
        assert (x.shape[1] == 2 and x.shape[4] == 3)

    # Bring image range from 0..255 to 0..1 and use floats (also, list[(2,H,W,3)] -> (batch_size,2,H,W,3))
    if use_mixed_precision is True:
        x_adapt = tf.cast(x, dtype=tf.float16)
    else:
        x_adapt = tf.cast(x, dtype=tf.float32)
    x_adapt /= 255.

    # Make sure the image dimensions are multiples of 2**pyramid_levels, pad them if they're not
    _, pad_h = divmod(int(x_adapt.shape[2]), 2 ** pyr_lvls)
    if pad_h != 0:
        pad_h = 2 ** pyr_lvls - pad_h
    _, pad_w = divmod(int(x_adapt.shape[3]), 2 ** pyr_lvls)
    if pad_w != 0:
        pad_w = 2 ** pyr_lvls - pad_w
    x_adapt_info = x_adapt.shape
    if pad_h != 0 or pad_w != 0:
        padding = [(0, 0), (0, 0), (0, pad_h), (0, pad_w), (0, 0)] 
        x_adapt_info = x_adapt.shape  # Save original shape
        x_adapt = tf.pad(x_adapt, padding, mode='constant', constant_values=0.)

    return x_adapt, x_adapt_info
      
def postproc_y_hat_test(y_hat, adapt_info=None):
    """Postprocess the results coming from the network during the test mode.
    Here, y_hat, is the actual data, not the y_hat TF tensor. Override as necessary.
    Args:
        y_hat: predictions, see set_output_tnsrs() for details
        adapt_info: adaptation information in (N,H,W,2) format
    Returns:
        Postprocessed labels
    """
    assert (isinstance(y_hat, list) and len(y_hat) == 2)

    # Have the samples been padded to fit the network's requirements? If so, crop flows back to original size.
    pred_flows = y_hat[0]
    if adapt_info is not None:
        pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]

    # Individuate flows of the flow pyramid (at this point, they are still batched)
    pyramids = y_hat[1]
    pred_flows_pyramid = []
    for idx in range(len(pred_flows)):
        pyramid = []
        for lvl in range(pyr_lvls - flow_pred_lvl + 1):
            pyramid.append(pyramids[lvl][idx])
        pred_flows_pyramid.append(pyramid)

    return pred_flows, pred_flows_pyramid

###
# PWC-Net pyramid helpers
###
def extract_features(x_tnsr, name='featpyr'):
    """Extract pyramid of features
    Args:
        x_tnsr: Input tensor (input pair of images in [batch_size, 2, H, W, 3] format)
        name: Variable scope name
    Returns:
        c1, c2: Feature pyramids
    Ref:
        Per page 3 of paper, section "Feature pyramid extractor," given two input images I1 and I2, we generate
        L-level pyramids of feature representations, with the bottom (zeroth) level being the input images,
        i.e., Ct<sup>0</sup> = It. To generate feature representation at the l-th layer, Ct<sup>l</sup>, we use
        layers of convolutional filters to downsample the features at the (l?1)th pyramid level, Ct<sup>l-1</sup>,
        by a factor of 2. From the first to the sixth levels, the number of feature channels are respectively
        16, 32, 64, 96, 128, and 196. Also see page 15 of paper for a rendering of the network architecture.
        Per page 15, individual images of the image pair are encoded using the same Siamese network. Each
        convolution is followed by a leaky ReLU unit. The convolutional layer and the x2 downsampling layer at
        each level is implemented using a single convolutional layer with a stride of 2.

        Note that Figure 4 on page 15 differs from the PyTorch implementation in two ways:
        - It's missing a convolution layer at the end of each conv block
        - It shows a number of filters of 192 (instead of 196) at the end of the last conv block

    """
    assert (1 <= pyr_lvls <= 6)
    if dbg:
        print(f"Building feature pyramids (c11,c21) ... (c1{pyr_lvls},c2{pyr_lvls})")
    # Make the feature pyramids 1-based for better readability down the line
    num_chann = [None, 16, 32, 64, 96, 128, 196]
    c1, c2 = [None], [None]
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        for pyr, x, reuse, name in zip([c1, c2], [x_tnsr[:, 0], x_tnsr[:, 1]], [None, True], ['c1', 'c2']):
            for lvl in range(1, pyr_lvls + 1):
                # tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name, reuse)
                # reuse is set to True because we want to learn a single set of weights for the pyramid
                # kernel_initializer = 'he_normal' or tf.keras.initializers.he_normal(seed=None)
                f = num_chann[lvl]
                x = tf.layers.conv2d(x, f, 3, 2, 'same', kernel_initializer=init, name=f'conv{lvl}a', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}a') # default alpha is 0.2 for TF
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}aa', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1)  # , name=f'relu{lvl+1}aa')
                x = tf.layers.conv2d(x, f, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}b', reuse=reuse)
                x = tf.nn.leaky_relu(x, alpha=0.1, name=f'{name}{lvl}')
                pyr.append(x)
    return c1, c2

###
# PWC-Net warping helpers
###
def warp(c2, sc_up_flow, lvl, name='warp'):
    """Warp a level of Image1's feature pyramid using the upsampled flow at level+1 of Image2's pyramid.
    Args:
        c2: The level of the feature pyramid of Image2 to warp
        sc_up_flow: Scaled and upsampled estimated optical flow (from Image1 to Image2) used for warping
        lvl: Index of that level
        name: Op scope name
    Ref:
        Per page 4 of paper, section "Warping layer," at the l-th level, we warp features of the second image toward
        the first image using the x2 upsampled flow from the l+1th level:
            C1w<sup>l</sup>(x) = C2<sup>l</sup>(x + Up2(w<sup>l+1</sup>)(x))
        where x is the pixel index and the upsampled flow Up2(w<sup>l+1</sup>) is set to be zero at the top level.
        We use bilinear interpolation to implement the warping operation and compute the gradients to the input
        CNN features and flow for backpropagation according to E. Ilg's FlowNet 2.0 paper.
        For non-translational motion, warping can compensate for some geometric distortions and put image patches
        at the right scale.

        Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
        and, hence, reduce the model size.
    """
    op_name = f'{name}{lvl}'
    if dbg:
        msg = f'Adding {op_name} with inputs {c2.op.name} and {sc_up_flow.op.name}'
        print(msg)
    with tf.name_scope(name):
        return dense_image_warp(c2, sc_up_flow, name=op_name)

def deconv(x, lvl, name='up_flow'):
    """Upsample, not using a bilinear filter, but rather learn the weights of a conv2d_transpose op filters.
    Args:
        x: Level features or flow to upsample
        lvl: Index of that level
        name: Op scope name
    """
    op_name = f'{name}{lvl}'
    if dbg:
        print(f'Adding {op_name} with input {x.op.name}')
    with tf.variable_scope('upsample'):
        # tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1), padding='valid', ... , name)
        return tf.layers.conv2d_transpose(x, 2, 4, 2, 'same', name=op_name)

###
# Cost Volume helpers
###
def corr(c1, warp, lvl, name='corr'):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: The level of the feature pyramid of Image1
        warp: The warped level of the feature pyramid of image22
        lvl: Index of that level
        name: Op scope name
    Ref:
        Per page 3 of paper, section "Cost Volume," a cost volume stores the data matching costs for associating
        a pixel from Image1 with its corresponding pixels in Image2. Most traditional optical flow techniques build
        the full cost volume at a single scale, which is both computationally expensive and memory intensive. By
        contrast, PWC-Net constructs a partial cost volume at multiple pyramid levels.

        The matching cost is implemented as the correlation between features of the first image and warped features
        of the second image:
            CV<sup>l</sup>(x1,x2) = (C1<sup>l</sup>(x1))<sup>T</sup> . Cw<sup>l</sup>(x2) / N
        where where T is the transpose operator and N is the length of the column vector C1<sup>l</sup>(x1).
        For an L-level pyramid, we only need to compute a partial cost volume with a limited search range of d
        pixels. A one-pixel motion at the top level corresponds to 2**(L?1) pixels at the full resolution images.
        Thus we can set d to be small, e.g. d=4. The dimension of the 3D cost volume is d**2 × Hl × Wl, where Hl
        and Wl denote the height and width of the L-th pyramid level, respectively.

        Per page 3 of paper, section "3. Approach," the warping and cost volume layers have no learnable parameters
        and, hence, reduce the model size.

        Per page 5 of paper, section "Implementation details," we use a search range of 4 pixels to compute the
        cost volume at each level.

    """
    op_name = f'corr{lvl}'
    if dbg:
        print(f'Adding {op_name} with inputs {c1.op.name} and {warp.op.name}')
    with tf.name_scope(name):
        return cost_volume(c1, warp, search_range, op_name)

###
# Optical flow estimator helpers
###
def predict_flow(corr, c1, up_flow, up_feat, lvl, name='predict_flow'):
    """Estimate optical flow.
    Args:
        corr: The cost volume at level lvl
        c1: The level of the feature pyramid of Image1
        up_flow: An upsampled version of the predicted flow from the previous level
        up_feat: An upsampled version of the features that were used to generate the flow prediction
        lvl: Index of the level
        name: Op scope name
    Args:
        upfeat: The features used to generate the predicted flow
        flow: The predicted flow
    Ref:
        Per page 4 of paper, section "Optical flow estimator," the optical flow estimator is a multi-layer CNN. Its
        input are the cost volume, features of the first image, and upsampled optical flow and its output is the
        flow w<sup>l</sup> at the l-th level. The numbers of feature channels at each convolutional layers are
        respectively 128, 128, 96, 64, and 32, which are kept fixed at all pyramid levels. The estimators at
        different levels have their own parameters instead of sharing the same parameters. This estimation process
        is repeated until the desired level, l0.

        Per page 5 of paper, section "Implementation details," we use a 7-level pyramid and set l0 to be 2, i.e.,
        our model outputs a quarter resolution optical flow and uses bilinear interpolation to obtain the
        full-resolution optical flow.

        The estimator architecture can be enhanced with DenseNet connections. The inputs to every convolutional
        layer are the output of and the input to its previous layer. DenseNet has more direct connections than
        traditional layers and leads to significant improvement in image classification.

        Note that we do not use DenseNet connections in this implementation because a) they increase the size of the
        model, and, b) per page 7 of paper, section "Optical flow estimator," removing the DenseNet connections
        results in higher training error but lower validation errors when the model is trained on FlyingChairs
        (that being said, after the model is fine-tuned on FlyingThings3D, DenseNet leads to lower errors).
    """
    op_name = f'flow{lvl}'
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        if c1 is None and up_flow is None and up_feat is None:
            if dbg:
                print(f'Adding {op_name} with input {corr.op.name}')
            x = corr
        else:
            if dbg:
                msg = f'Adding {op_name} with inputs {corr.op.name}, {c1.op.name}, {up_flow.op.name}, {up_feat.op.name}'
                print(msg)
            x = tf.concat([corr, c1, up_flow, up_feat], axis=3)

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_0')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_1')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_2')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_3')
        act = tf.nn.leaky_relu(conv, alpha=0.1)
        x = tf.concat([act, x], axis=3) if use_dense_cx else act

        conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name=f'conv{lvl}_4')
        act = tf.nn.leaky_relu(conv, alpha=0.1)  # will also be used as an input by the context network
        upfeat = tf.concat([act, x], axis=3, name=f'upfeat{lvl}') if use_dense_cx else act

        flow = tf.layers.conv2d(upfeat, 2, 3, 1, 'same', name=op_name)

        return upfeat, flow

###
# PWC-Net context network helpers
###
def refine_flow(feat, flow, lvl, name='ctxt'):
    """Post-ptrocess the estimated optical flow using a "context" nn.
    Args:
        feat: Features of the second-to-last layer from the optical flow estimator
        flow: Estimated flow to refine
        lvl: Index of the level
        name: Op scope name
    Ref:
        Per page 4 of paper, section "Context network," traditional flow methods often use contextual information
        to post-process the flow. Thus we employ a sub-network, called the context network, to effectively enlarge
        the receptive field size of each output unit at the desired pyramid level. It takes the estimated flow and
        features of the second last layer from the optical flow estimator and outputs a refined flow.

        The context network is a feed-forward CNN and its design is based on dilated convolutions. It consists of
        7 convolutional layers. The spatial kernel for each convolutional layer is 3×3. These layers have different
        dilation constants. A convolutional layer with a dilation constant k means that an input unit to a filter
        in the layer are k-unit apart from the other input units to the filter in the layer, both in vertical and
        horizontal directions. Convolutional layers with large dilation constants enlarge the receptive field of
        each output unit without incurring a large computational burden. From bottom to top, the dilation constants
        are 1, 2, 4, 8, 16, 1, and 1.
    """
    op_name = f'refined_flow{lvl}'
    if dbg:
        print(f'Adding {op_name} sum of dc_convs_chain({feat.op.name}) with {flow.op.name}')
    init = tf.keras.initializers.he_normal()
    with tf.variable_scope(name):
        x = tf.layers.conv2d(feat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}1')
        x = tf.nn.leaky_relu(x, alpha=0.1)  # default alpha is 0.2 for TF
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name=f'dc_conv{lvl}2')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name=f'dc_conv{lvl}3')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name=f'dc_conv{lvl}4')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name=f'dc_conv{lvl}5')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}6')
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name=f'dc_conv{lvl}7')

        return tf.add(flow, x, name=op_name)


###
# PWC-Net nn builder
###
#TODO Add reuse_variables for forward flow
def pwcnet(x_tnsr, name='pwcnet', reuse=False):
    """Defines and connects the backbone neural nets
    Args:
        inputs: TF placeholder that contains the input frame pairs in [batch_size, 2, H, W, 3] format
        name: Name of the nn
    Returns:
        net: Output tensors of the backbone network
    Ref:
        RE: the scaling of the upsampled estimated optical flow, per page 5, section "Implementation details," we
        do not further scale the supervision signal at each level, the same as the FlowNet paper. As a result, we
        need to scale the upsampled flow at each pyramid level for the warping layer. For example, at the second
        level, we scale the upsampled flow from the third level by a factor of 5 (=20/4) before warping features
        of the second image.
    Based on:
        - https://github.com/daigo0927/PWC-Net_tf/blob/master/model.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    with tf.variable_scope(name) as scope:
        #reeki
        if reuse:
            scope.reuse_variables()

        # Extract pyramids of CNN features from both input images (1-based lists))
        c1, c2 = extract_features(x_tnsr)

        flow_pyr = []

        for lvl in range(pyr_lvls, flow_pred_lvl - 1, -1):

            if lvl == pyr_lvls:
                # Compute the cost volume
                corr_val = corr(c1[lvl], c2[lvl], lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr_val, None, None, None, lvl)
            else:
                # Warp level of Image1's using the upsampled flow
                scaler = 20. / 2 ** lvl  # scaler values are 0.625, 1.25, 2.5, 5.0
                warp_val = warp(c2[lvl], up_flow * scaler, lvl)

                # Compute the cost volume
                corr_val = corr(c1[lvl], warp_val, lvl)

                # Estimate the optical flow
                upfeat, flow = predict_flow(corr_val, c1[lvl], up_flow, up_feat, lvl)

            _, lvl_height, lvl_width, _ = tf.unstack(tf.shape(c1[lvl]))

            if lvl != flow_pred_lvl:
                if use_res_cx:
                    flow = refine_flow(upfeat, flow, lvl)

                # Upsample predicted flow and the features used to compute predicted flow
                flow_pyr.append(flow)

                up_flow = deconv(flow, lvl, 'up_flow')
                up_feat = deconv(upfeat, lvl, 'up_feat')
            else:
                # Refine the final predicted flow
                flow = refine_flow(upfeat, flow, lvl)
                flow_pyr.append(flow)

                # Upsample the predicted flow (final output) to match the size of the images
                scaler = 2 ** flow_pred_lvl
                if dbg:
                    print(f'Upsampling {flow.op.name} by {scaler} in each dimension.')
                size = (lvl_height * scaler, lvl_width * scaler)
                flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scaler
                break

        return flow_pred, flow_pyr
