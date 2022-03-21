import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

def compat_batch_norm(x, activation_fn=None, scope=None):
    with tf.compat.forward_compatibility_horizon(2019, 5, 1):
        # print("*******econet compat_batch_norm**********")
        return slim.batch_norm(x, activation_fn=activation_fn, scope=scope)

def ECONet(inputs, 
           opt=None, 
           final_endpoint='Mixed_5c', 
           scope='InceptionV1', 
           is_training=True, 
           reuse=False):

    # Init Setting
    batch_norm_params = {
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }

    activation_fn = tf.nn.relu
    prediction_fn = slim.softmax
    # use_batch_norm = True 
    weight_decay = opt['weight_decay']
    net2d_keep_prob = opt['net2d_keep_prob']
    net3d_keep_prob = opt['net3d_keep_prob']
    num_segments = opt['num_segments']
    num_classes = opt['num_classes']
    with tf.variable_scope(scope, 'ECONet', [inputs], reuse=reuse):
        
        with tf.variable_scope('2DNet'):

            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=None,
                normalizer_fn=None
                ):               

                with slim.arg_scope([slim.batch_norm], **batch_norm_params):

                    end_points = {}
                    # (num_segments*batchsize) x 224 x 224 x 3
                    # 224 x 224 x 3

                    # print(inputs)
                    end_point = 'conv1_7x7_s2'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]])
                        inputs = tf.pad(inputs, paddings, 'CONSTANT') # Padding with zero value
                        net2d = slim.conv2d(inputs, 64, [7, 7], stride=2, padding='VALID', scope=end_point)
                        net2d = compat_batch_norm(net2d, activation_fn=activation_fn, scope=end_point+'_bn')
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points


                    end_point = 'pool1_3x3_s2'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        net2d = tf.pad(net2d, paddings, 'CONSTANT') # Padding with zero value
                        net2d = slim.max_pool2d(net2d, [3, 3], stride=2, scope=end_point)
                        # net2d = tf.nn.max_pool2d(net2d, [3, 3], stride=2, scope=end_point)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points  
                        

                    end_point = 'conv2_3x3'        
                    with tf.variable_scope(end_point):
                        net2d = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='reduce')
                        net2d = compat_batch_norm(net2d, activation_fn=activation_fn, scope='reduce'+'_bn')
                        net2d = slim.conv2d(net2d, 192, [3, 3], padding='SAME', scope='conv2_3x3')
                        net2d = compat_batch_norm(net2d, activation_fn=activation_fn, scope='conv2_3x3'+'_bn')
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'pool2_3x3_s2'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        net2d = tf.pad(net2d, paddings, 'CONSTANT') # Padding with zero value
                        net2d = slim.max_pool2d(net2d, [3, 3], stride=2, scope=end_point)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points  
                    

                    end_point = 'inception_3a'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 64, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 96, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 96, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 32, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points  


                    end_point = 'inception_3b'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 96, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 96, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 96, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 64, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points  


                    end_point = 'inception_3c'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        branch1 = slim.conv2d(net2d, 128, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = tf.pad(branch1, paddings, 'CONSTANT') # Padding with zero value
                        branch1 = slim.conv2d(branch1, 160, [3, 3], stride=2, padding='VALID', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')

                        net3d = slim.conv2d(branch2, 96, [3, 3], padding='SAME', scope='_double_3x3_1') # Output features for 3DResNet
                        net3d = compat_batch_norm(net3d, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')

                        branch2 = tf.pad(net3d, paddings, 'CONSTANT') # Padding with zero value
                        branch2 = slim.conv2d(branch2, 96, [3, 3], stride=2, padding='VALID', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        net2d = tf.pad(net2d, paddings, 'CONSTANT') # Padding with zero value
                        branch3 = slim.max_pool2d(net2d, [3, 3], stride=2, padding='VALID', scope='_pool')
                        net2d = tf.concat(values=[branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'inception_4a'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 224, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 64, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 96, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 96, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 128, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 128, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points  


                    end_point = 'inception_4b'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 192, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 96, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 128, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 96, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 128, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 128, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'inception_4c'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 160, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 128, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 160, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 128, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 160, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 160, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'inception_4d'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 96, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 128, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 192, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 160, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 192, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 192, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')                
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points

                    end_point = 'inception_4e'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        branch1 = slim.conv2d(net2d, 128, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = tf.pad(branch1, paddings, 'CONSTANT') # Padding with zero value
                        branch1 = slim.conv2d(branch1, 192, [3, 3], stride=2, padding='VALID', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 192, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 256, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = tf.pad(branch2, paddings, 'CONSTANT') # Padding with zero value
                        branch2 = slim.conv2d(branch2, 256, [3, 3], stride=2, padding='VALID', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                        net2d = tf.pad(net2d, paddings, 'CONSTANT') # Padding with zero value
                        branch3 = slim.max_pool2d(net2d, [3, 3], stride=2, padding='VALID', scope='_pool')
                        net2d = tf.concat(values=[branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'inception_5a'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 352, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 192, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 320, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 160, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 224, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 224, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.avg_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 


                    end_point = 'inception_5b'
                    with tf.variable_scope(end_point):
                        branch0 = slim.conv2d(net2d, 352, [1, 1], padding='SAME', scope='_1x1')
                        branch0 = compat_batch_norm(branch0, activation_fn=activation_fn, scope='_1x1'+'_bn')
                        branch1 = slim.conv2d(net2d, 192, [1, 1], padding='SAME', scope='_3x3_reduce')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3_reduce'+'_bn')
                        branch1 = slim.conv2d(branch1, 320, [3, 3], padding='SAME', scope='_3x3')
                        branch1 = compat_batch_norm(branch1, activation_fn=activation_fn, scope='_3x3'+'_bn')
                        branch2 = slim.conv2d(net2d, 192, [1, 1], padding='SAME', scope='_double_3x3_reduce')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_reduce'+'_bn')
                        branch2 = slim.conv2d(branch2, 224, [3, 3], padding='SAME', scope='_double_3x3_1')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_1'+'_bn')
                        branch2 = slim.conv2d(branch2, 224, [3, 3], padding='SAME', scope='_double_3x3_2')
                        branch2 = compat_batch_norm(branch2, activation_fn=activation_fn, scope='_double_3x3_2'+'_bn')
                        branch3 = slim.max_pool2d(net2d, [3, 3], stride=1, padding='SAME', scope='_pool')
                        branch3 = slim.conv2d(branch3, 128, [1, 1], padding='SAME', scope='_pool_proj')
                        branch3 = compat_batch_norm(branch3, activation_fn=activation_fn, scope='_pool_proj'+'_bn')
                        net2d = tf.concat(values=[branch0, branch1, branch2, branch3], axis=3)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 

                    end_point = 'global_pool2d'
                    # 1st Global average pooling.
                    # net2d = tf.reduce_mean(
                    #         input_tensor=net2d, axis=[1, 2], name=end_point, keepdims=True)   
                    with tf.variable_scope(end_point):
                        net2d = slim.avg_pool2d(net2d, [7, 7], stride=1, padding='VALID', scope=end_point)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points 

                    end_point = 'global_pool2d_drop'
                    with tf.variable_scope(end_point):
                        net2d = slim.dropout(net2d, keep_prob=net2d_keep_prob, is_training=is_training, scope=end_point)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points

                    # print(net2d.shape) # B*N x 1 x 1 x 1024

                    end_point = 'global_pool2d_reshape_consensus'
                    with tf.variable_scope(end_point):
                        net2d = tf.reshape(net2d, [-1, num_segments, net2d.shape[1], net2d.shape[2], net2d.shape[3]])
                        net2d = slim.avg_pool3d(net2d, [num_segments, 1, 1], stride=1, scope=end_point)
                        end_points[end_point] = net2d
                        if end_point == final_endpoint: return net2d, end_points

                    # print(net2d.shape) # B x 1 x 1 x 1 x 1024

        with tf.variable_scope('3DNet'):

            with slim.arg_scope(
                [slim.conv3d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=None,
                normalizer_fn=None,
                ):

                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    
                    end_point = 'res3'
                    with tf.variable_scope(end_point):
                        debug_net3d = net3d
                        net3d = tf.reshape(net3d, [-1, num_segments, net3d.shape[1], net3d.shape[2], net3d.shape[3]])
                        shortcut = slim.conv3d(net3d, 128, [3, 3, 3], padding='SAME', scope='a_1')
                        residual = compat_batch_norm(shortcut, activation_fn=activation_fn, scope='a_1_bn')
                        # residual = shortcut

                        residual = slim.conv3d(residual, 128, [3, 3, 3], padding='SAME', scope='b_1')
                        residual = compat_batch_norm(residual, activation_fn=activation_fn, scope='b_1_bn')
                        residual = slim.conv3d(residual, 128, [3, 3, 3], padding='SAME', scope='b_2')
                        net3d = residual + shortcut
                        net3d = compat_batch_norm(net3d, activation_fn=activation_fn, scope='b_2_bn')
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points
                    

                    end_point = 'res4'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
                        residual = tf.pad(net3d, paddings, 'CONSTANT') # Padding with zero value
                        residual = slim.conv3d(residual, 256, [3, 3, 3], stride=[2, 2, 2],
                                            padding='VALID', scope='a_1')
                        residual = compat_batch_norm(residual, activation_fn=activation_fn, scope='a_1_bn')
                        residual = slim.conv3d(residual, 256, [3, 3, 3], padding='SAME', scope='a_2')
                        shortcut = tf.pad(net3d, paddings, 'CONSTANT')
                        shortcut = slim.conv3d(shortcut, 256, [3, 3, 3], stride=[2, 2, 2], \
                                            padding='VALID', scope='a_down')
                        net3d = residual + shortcut
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points

                    # print(net3d.shape)

                    end_point = 'res5'
                    with tf.variable_scope(end_point):
                        shortcut = net3d
                        residual = compat_batch_norm(shortcut, activation_fn=activation_fn, scope='a_1_bn')
                        # residual = shortcut

                        residual = slim.conv3d(residual, 256, [3, 3, 3], padding='SAME', scope='b_1')
                        residual = compat_batch_norm(residual, activation_fn=activation_fn, scope='b_1_bn')
                        residual = slim.conv3d(residual, 256, [3, 3, 3], padding='SAME', scope='b_2')
                        net3d = residual + shortcut
                        net3d = compat_batch_norm(net3d, activation_fn=activation_fn, scope='b_2_bn')
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points

                    end_point = 'res6'
                    with tf.variable_scope(end_point):
                        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
                        residual = tf.pad(net3d, paddings, 'CONSTANT') # Padding with zero value
                        residual = slim.conv3d(residual, 512, [3, 3, 3], stride=[2, 2, 2],
                                            padding='VALID', scope='a_1')
                        residual = compat_batch_norm(residual, activation_fn=activation_fn, scope='a_1_bn')
                        residual = slim.conv3d(residual, 512, [3, 3, 3], padding='SAME', scope='a_2')
                        shortcut = tf.pad(net3d, paddings, 'CONSTANT')
                        shortcut = slim.conv3d(shortcut, 512, [3, 3, 3], stride=[2, 2, 2], \
                                            padding='VALID', scope='a_down')
                        net3d = residual + shortcut
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points

                    end_point = 'res7'
                    with tf.variable_scope(end_point):
                        shortcut = net3d
                        residual = compat_batch_norm(shortcut, activation_fn=activation_fn, scope='a_1_bn')
                        # residual = shortcut
                        residual = slim.conv3d(residual, 512, [3, 3, 3], padding='SAME', scope='b_1')
                        residual = compat_batch_norm(residual, activation_fn=activation_fn, scope='b_1_bn')
                        residual = slim.conv3d(residual, 512, [3, 3, 3], padding='SAME', scope='b_2')
                        net3d = residual + shortcut
                        net3d = compat_batch_norm(net3d, activation_fn=activation_fn, scope='b_2_bn')
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points

                    end_point = 'global_pool3d'
                    with tf.variable_scope(end_point):
                        net3d = slim.avg_pool3d(net3d, [math.ceil(num_segments/4), 7, 7], stride=1, scope=end_point)
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points   
                    # print(net3d.shape, math.ceil(num_segments/4))

                    end_point = 'global_pool3d_drop'
                    with tf.variable_scope(end_point):
                        net3d = slim.dropout(net3d, keep_prob=net3d_keep_prob, is_training=is_training, scope=end_point)
                        end_points[end_point] = net3d
                        if end_point == final_endpoint: return net3d, end_points

        with tf.variable_scope('Fusion'):
            # TODO
            # net2d = net2d[:16, :, :, :]

            # for debug TODO
            net = tf.concat(values=[net2d, net3d], axis=-1)
            net = tf.squeeze(net, axis=[1, 2, 3])
            # net = tf.squeeze(net2d, axis=[1, 2, 3])

            # print(net.shape)
            # net = tf.squeeze(net, axis=[2])

            logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc') # TODO
            end_points['logits'] = logits
            end_points['predictions'] = prediction_fn(logits, scope='predictions')


    return logits, end_points