import keras
from keras import layers
from keras import models

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _conv_block(inputs, strides, filters, kernel=3):
    """
    Adds an initial convolution layer (with batch normalization and relu6).
    """
    x = layers.Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='Conv1')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv1_bn')(x)
    
    print(x.name, inputs.shape, x.shape)
    
    return layers.ReLU(6., name='Conv1_relu')(x)

def _sep_conv_block(inputs, filters, alpha, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1)):
    """
    Adds a separable convolution block.
    A separable convolution block consists of a depthwise conv,
    and a pointwise convolution.
    """
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='Dw_conv_sep')(inputs)
    x = layers.Conv2D(pointwise_conv_filters, (1, 1), padding='valid', use_bias=False, 
                      strides=strides, name='Conv_sep')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_sep_bn')(x)
    
    print(x.name, inputs.shape, x.shape)
    
    return layers.ReLU(6., name='Conv_sep_relu')(x)
    
def _inverted_res_block(inputs, kernel, expansion, alpha, filters, block_id, stride=1):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    
    if block_id:
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_bn')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    x = layers.DepthwiseConv2D(kernel_size=kernel,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_bn')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_bn')(x)

    print(x.name, inputs.shape, x.shape)
    
    if in_channels == pointwise_filters and stride == 1:
        print("Adding %s" % x.name)
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x
    
def MnasNet(input_shape=None, alpha=1.0, depth_multiplier=1, pooling=None, nb_classes=10):
    img_input = layers.Input(shape=input_shape)
    
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(img_input, strides=2, filters=first_block_filters)

    x = _sep_conv_block(x, filters=16, alpha=alpha, 
                        pointwise_conv_filters=16, depth_multiplier=depth_multiplier)
    
    x = _inverted_res_block(x, kernel=3, expansion=3, stride=2, alpha=alpha, filters=24, block_id=1)
    x = _inverted_res_block(x, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=2)
    x = _inverted_res_block(x, kernel=3, expansion=3, stride=1, alpha=alpha, filters=24, block_id=3)
    
    x = _inverted_res_block(x, kernel=5, expansion=3, stride=2, alpha=alpha, filters=40, block_id=4)
    x = _inverted_res_block(x, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=5)
    x = _inverted_res_block(x, kernel=5, expansion=3, stride=1, alpha=alpha, filters=40, block_id=6)
    
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=80, block_id=7)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=8)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=80, block_id=9)

    x = _inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=10)
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=96, block_id=11)
    
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=2, alpha=alpha, filters=192, block_id=12)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=13)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=14)
    x = _inverted_res_block(x, kernel=5, expansion=6, stride=1, alpha=alpha, filters=192, block_id=15)
    
    x = _inverted_res_block(x, kernel=3, expansion=6, stride=1, alpha=alpha, filters=320, block_id=16)
    
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)
        
    x = layers.Dense(nb_classes, activation='softmax', use_bias=True, name='proba')(x)
    
    inputs = img_input
    
    model = models.Model(inputs, x, name='mnasnet')
    return model