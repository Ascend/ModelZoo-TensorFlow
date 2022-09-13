import math
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import ops
import npu_device


# @ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op, grad):
    """ The gradient for fastgelu

    Args:
        op:The fastgelu operations that we are differentiating,which we can us to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the fast_gelu op.

    Returns:
        Gradient with respect to the input of fast_gelu
    """
    return [npu_device.gen_npu_ops.fast_gelu_grad(grad, op.inputs[0])]


grad_registry_list = ops.gradient_registry.list()
if not hasattr(npu_device.ops,
               'gelu') and "FastGelu" not in grad_registry_list:
    ops.RegisterGradient("FastGelu")(_fast_gelu_grad)


@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.

    Returns:
        `x` with the GELU activation applied.
    """
    if not hasattr(npu_device.ops, 'gelu'):
        return npu_device.gen_npu_ops.fast_gelu(x)
    else:
        fast_gelu = getattr(npu_device.ops, 'gelu')
        return fast_gelu(x)


# default
# def gelu(x):
#     """An approximation of gelu.

#     See: https://arxiv.org/pdf/1606.08415.pdf
#     """
#     return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
