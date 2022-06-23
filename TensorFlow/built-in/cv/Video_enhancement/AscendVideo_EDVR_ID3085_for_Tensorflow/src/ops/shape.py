import tensorflow as tf


def get_tensor_shape(x, dim=None, allow_convert_to_int=True):
    shape_tensor = tf.shape(x)

    ndim = shape_tensor.get_shape().as_list()[0]
    shapes_ = tf.split(shape_tensor, ndim)

    shapes = [tf.squeeze(s) for s in shapes_]

    if allow_convert_to_int:
        shape_int_list = x.get_shape().as_list()
        for i in range(ndim):
            if shape_int_list[i] is not None:
                shapes[i] = shape_int_list[i]

    if dim is None:
        shape_tensor_as_list = [shapes[d] for d in range(ndim)]
    elif isinstance(dim, (tuple, list)):
        shape_tensor_as_list = [shapes[d] for d in dim]
    else:
        shape_tensor_as_list = shapes[dim]

    return shape_tensor_as_list
