import math
import numpy as np

#https://github.com/tensorflow/tensorflow/blob/r2.8/tensorflow/python/keras/layers/core.py
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
from tensorflow.python.util.tf_export import keras_export
class SzDense(Layer):
  def __init__(self,
               units,
               halfbandwidth=0, 
               param_reduction=0.5, 
               form='diagonal', 
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(SzDense, self).__init__(
        activity_regularizer=activity_regularizer, **kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    if self.units < 0:
      raise ValueError(f'Received an invalid value for `units`, expected '
                       f'a positive integer, got {units}.')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.input_spec = InputSpec(min_ndim=2)
    self.supports_masking = True

    self.halfbandwidth = halfbandwidth
    self.form = form
    self.reduction_sv = param_reduction
    self.num_ones = 0
    self.reduced_ratio = 0
    self.num_weights = 0
    self.reduced_ratio = 0

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight(
        'kernel',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    self.window = self.add_weight(name='window',
                                  shape=[last_dim, self.units],
                                  initializer='ones',
                                  trainable=False)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None

    #window init
    self.num_ones = 0
    self.reduced_ratio = 0
    nx = last_dim
    ny = self.units
    self.num_weights = nx * ny
    if self.halfbandwidth == 0:
      self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
      if self.form == 'gaussian':
        self.halfbandwidth *= 1.5
    #endif
    wnd = np.zeros((nx,ny))
    w_corr = 1.
    if self.form == 'diagonal':
      if ny > 1:
        rxy = (nx-1) / (ny-1)
        hwdiv = self.halfbandwidth * math.sqrt(rxy * rxy + 1)
        for iy in range(ny):
          ix1 = rxy * iy - hwdiv
          ix1 = int(ix1) + 1 if ix1 >= 0 else 0
          if ix1 > nx-1:
            continue
          ix2 = rxy * iy + hwdiv
          ix2 = math.ceil(ix2) if ix2 < nx else nx
          wnd[ix1:ix2, iy:iy+1] = 1
          self.num_ones += (ix2-ix1)
        #for ixiy
      else:
        wnd[:,:] = 1
        self.num_ones += nx
      #endif ny>1
      self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
      if self.num_ones > 0:
        w_corr = self.num_weights / self.num_ones
      self.kernel.assign(self.kernel * (wnd * w_corr))
    elif self.form == 'gaussian':
      if (self.halfbandwidth > 0) and (ny > 1):
        sgm2 = 1. / (2. * self.halfbandwidth * self.halfbandwidth)
        gsum = 0
        rxy = (nx-1) / (ny-1)
        for ix in range(nx):
          for iy in range(ny):
            gauss = math.exp(-(ix-rxy*iy)*(ix-rxy*iy)*sgm2)
            wnd[ix][iy] = gauss
            gsum += gauss
        #for ixiy
        self.reduced_ratio = 1. - gsum / self.num_weights
        if gsum > 0:
          w_corr = self.num_weights / gsum
        wnd = wnd * w_corr
      else:
        wnd[:,:] = 1
        self.num_ones = nx * ny
      #endif halfbandwidth
    elif self.form == 'random':
      wnd = np.random.rand(nx,ny)
      wnd = np.where(wnd < self.reduction_sv, 0, 1)
      self.num_ones = np.sum(wnd)
      self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
      if self.num_ones > 0:
        w_corr = self.num_weights / self.num_ones
      self.kernel.assign(self.kernel * (wnd * w_corr))
    #endif form_function
    self.window.assign(wnd)

    self.built = True

  def call(self, inputs):
    if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
      inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

    rank = inputs.shape.rank
    if rank == 2 or rank is None:
      # We use embedding_lookup_sparse as a more efficient matmul operation for
      # large sparse input tensors. The op will result in a sparse gradient, as
      # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
      # gradients. This can lead to sigfinicant speedups, see b/171762937.
      if isinstance(inputs, sparse_tensor.SparseTensor):
        # We need to fill empty rows, as the op assumes at least one id per row.
        inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
        # We need to do some munging of our input to use the embedding lookup as
        # a matrix multiply. We split our input matrix into separate ids and
        # weights tensors. The values of the ids tensor should be the column
        # indices of our input matrix and the values of the weights tensor
        # can continue to the actual matrix weights.
        # The column arrangement of ids and weights
        # will be summed over and does not matter. See the documentation for
        # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
        # of the inputs to both ops.
        ids = sparse_tensor.SparseTensor(
            indices=inputs.indices,
            values=inputs.indices[:, 1],
            dense_shape=inputs.dense_shape)
        weights = inputs
        outputs = embedding_ops.embedding_lookup_sparse_v2(
            self.kernel * self.window, ids, weights, combiner='sum')
      else:
        outputs = gen_math_ops.MatMul(a=inputs, b=self.kernel * self.window)
    # Broadcast kernel to inputs.
    else:
      outputs = standard_ops.tensordot(inputs, self.kernel * self.window, [[rank - 1], [0]])
      # Reshape the output back to the original ndim of the input.
      if not context.executing_eagerly():
        shape = inputs.shape.as_list()
        output_shape = shape[:-1] + [self.kernel.shape[-1]]
        outputs.set_shape(output_shape)

    if self.use_bias:
      outputs = nn_ops.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % (input_shape,))
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    })
    return config
  def get_num_zeros(self):
    return(self.num_weights - self.num_ones)
  def get_num_weights(self):
    return(self.num_weights)
  def get_reduced_ratio(self):
    return(self.reduced_ratio)
  def get_halfbandwidth(self):
    return(self.halfbandwidth)
#class SzDense

#https://github.com/tensorflow/tensorflow/blob/r2.8/tensorflow/python/keras/layers/convolutional.py
import functools

class SzConv(Layer):
  def __init__(self,
               rank,
               filters,
               kernel_size,
               param_reduction=0.5, 
               form='individual', 
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               conv_op=None,
               **kwargs):
    super(SzConv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank

    if isinstance(filters, float):
      filters = int(filters)
    if filters is not None and filters < 0:
      raise ValueError(f'Received a negative value for `filters`.'
                       f'Was expecting a positive value, got {filters}.')
    self.filters = filters
    self.groups = groups or 1
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')

    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=self.rank + 2)

    self._validate_init()
    self._is_causal = self.padding == 'causal'
    self._channels_first = self.data_format == 'channels_first'
    self._tf_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)

    self.reduction_sv = param_reduction
    self.form = form
    self.num_ones = 0
    self.num_weights = 0
    self.reduced_ratio = 0
    self.halfbandwidth = 0

  def _validate_init(self):
    if self.filters is not None and self.filters % self.groups != 0:
      raise ValueError(
          'The number of filters must be evenly divisible by the number of '
          'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

    if not all(self.strides):
      raise ValueError('The argument `strides` cannot contains 0(s). '
                       'Received: %s' % (self.strides,))

    if (self.padding == 'causal' and not isinstance(self,
                                                    (SzConv1D, SzSeparableConv1D))):
      raise ValueError('Causal padding is only supported for `SzConv1D`'
                       'and `SzSeparableConv1D`.')

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))
    kernel_shape = self.kernel_size + (input_channel // self.groups,
                                       self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    self.window = self.add_weight(name='window', 
                                  shape=kernel_shape, 
                                  initializer='ones', 
                                  trainable=False)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, str):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    tf_op_name = self.__class__.__name__
    if tf_op_name == 'SzConv1D':
      tf_op_name = 'conv1d'  # Backwards compat.

    self._convolution_op = functools.partial(
        nn_ops.convolution_v2,
        strides=tf_strides,
        padding=tf_padding,
        dilations=tf_dilations,
        data_format=self._tf_data_format,
        name=tf_op_name)

    #window initialization
    wnd = np.zeros(kernel_shape)
    w_corr = 1.
    nx = input_channel // self.groups
    ny = self.filters
    if self.form == 'individual':
      wnd = np.random.random_sample(kernel_shape)
      wnd = np.where(wnd < self.reduction_sv, 0, 1)
    elif self.form == 'kernel':
      for ix in range(nx):
        for iy in range(ny):
          if random.random() > self.reduction_sv:
            wnd[..., ix, iy] = 1
    elif self.form == 'diagonal':
      self.halfbandwidth = (nx*ny / math.sqrt(nx*nx + ny*ny)) * (1. - math.sqrt(self.reduction_sv)) 
      if ny > 1:
        rxy = (nx-1) / (ny-1)
        hwdiv = self.halfbandwidth * math.sqrt(rxy * rxy + 1)
        for iy in range(ny):
          ix1 = rxy * iy - hwdiv
          ix1 = int(ix1) + 1 if ix1 >= 0 else 0
          if ix1 > nx-1:
            continue
          ix2 = rxy * iy + hwdiv
          ix2 = math.ceil(ix2) if ix2 < nx else nx
          wnd[..., ix1:ix2, iy:iy+1] = 1
        #for ixiy
      else:
        wnd = np.ones(kernel_shape)
      #endif ny>1
    #endif self.form
    self.num_ones = np.sum(wnd)
    self.num_weights = wnd.size
    self.reduced_ratio = (self.num_weights - self.num_ones) / self.num_weights
    if self.num_ones > 0:
      w_corr = self.num_weights / self.num_ones
    self.kernel.assign(self.kernel * (wnd * w_corr))
    self.window.assign(wnd)

    self.built = True

  def call(self, inputs):
    input_shape = inputs.shape

    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    outputs = self._convolution_op(inputs, self.kernel * self.window)

    if self.use_bias:
      output_rank = outputs.shape.rank
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

          outputs = conv_utils.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = nn.bias_add(
              outputs, self.bias, data_format=self._tf_data_format)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_rank = len(input_shape) - self.rank - 1
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape(
          input_shape[:batch_rank]
          + self._spatial_output_shape(input_shape[batch_rank:-1])
          + [self.filters])
    else:
      return tensor_shape.TensorShape(
          input_shape[:batch_rank] + [self.filters] +
          self._spatial_output_shape(input_shape[batch_rank + 1:]))

  def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
    return False

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'dilation_rate':
            self.dilation_rate,
        'groups':
            self.groups,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if getattr(inputs.shape, 'ndims', None) is None:
      batch_rank = 1
    else:
      batch_rank = len(inputs.shape) - 2
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return -1 - self.rank
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding
  def get_num_zeros(self):
    return(self.num_weights - self.num_ones)
  def get_num_weights(self):
    return(self.num_weights)
  def get_reduced_ratio(self):
    return(self.reduced_ratio)
  def get_halfbandwidth(self):
    return(self.halfbandwidth)
#class SzConv

class SzConv2D(SzConv):
 def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(SzConv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=groups,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
#class SzConv2D

