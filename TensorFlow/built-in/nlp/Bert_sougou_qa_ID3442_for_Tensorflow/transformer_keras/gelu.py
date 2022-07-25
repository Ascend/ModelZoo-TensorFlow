import math

from tensorflow.keras import backend as K
from npu_bridge.npu_init import *


def gelu(x):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    #return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    return npu_unary_ops.gelu(x)
