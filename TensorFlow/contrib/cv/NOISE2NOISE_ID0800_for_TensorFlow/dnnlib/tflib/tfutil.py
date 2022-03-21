# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Miscellaneous helper utils for Tensorflow."""

import numpy as np
import tensorflow as tf
from npu_bridge.npu_init import *
from typing import Any, Iterable, List, Union

TfExpression = Union[tf.Tensor, tf.Variable, tf.Operation]
"""A type that represents a valid Tensorflow expression."""

TfExpressionEx = Union[TfExpression, int, float, np.ndarray]
"""A type that can be converted to a valid Tensorflow expression."""


def run(*args, **kwargs) -> Any:
    """Run the specified ops in the default session."""
    assert_tf_initialized()
    return tf.get_default_session().run(*args, **kwargs)


def is_tf_expression(x: Any) -> bool:
    """Check whether the input is a valid Tensorflow expression, i.e., Tensorflow Tensor, Variable, or Operation."""
    return isinstance(x, (tf.Tensor, tf.Variable, tf.Operation))


def shape_to_list(shape: Iterable[tf.Dimension]) -> List[Union[int, None]]:
    """Convert a Tensorflow shape to a list of ints."""
    return [dim.value for dim in shape]


def flatten(x: TfExpressionEx) -> TfExpression:
    """Shortcut function for flattening a tensor."""
    with tf.name_scope("Flatten"):
        return tf.reshape(x, [-1])


def log2(x: TfExpressionEx) -> TfExpression:
    """Logarithm in base 2."""
    with tf.name_scope("Log2"):
        return tf.log(x) * np.float32(1.0 / np.log(2.0))


def exp2(x: TfExpressionEx) -> TfExpression:
    """Exponent in base 2."""
    with tf.name_scope("Exp2"):
        return tf.exp(x * np.float32(np.log(2.0)))


def lerp(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpressionEx:
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t


def lerp_clip(a: TfExpressionEx, b: TfExpressionEx, t: TfExpressionEx) -> TfExpression:
    """Linear interpolation with clip."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def absolute_name_scope(scope: str) -> tf.name_scope:
    """Forcefully enter the specified name scope, ignoring any surrounding scopes."""
    return tf.name_scope(scope + "/")


def init_tf(config_dict: dict = None, graph=None) -> tf.Session:
    """Initialize TensorFlow session using good default settings."""
    if tf.get_default_session() is None:
        tf.set_random_seed(np.random.randint(1 << 31))
        return create_session(config_dict, force_as_default=True, graph=graph)


def assert_tf_initialized():
    """Check that TensorFlow session has been initialized."""
    if tf.get_default_session() is None:
        raise RuntimeError("No default TensorFlow session found. Please call dnnlib.tflib.tfutil.init_tf().")


def create_session(config_dict: dict = None, force_as_default: bool = False, graph=None) -> tf.Session:
    """Create tf.Session based on config dict."""
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()

    print('tf_config_dict:{}'.format(config_dict))

    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = config_dict["use_off_line"]
    custom_op.parameter_map["enable_data_pre_proc"].b = config_dict["enable_data_pre_proc"]
    custom_op.parameter_map["mix_compile_mode"].b = config_dict["mix_compile_mode"]
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(config_dict["precision_mode"])
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes(config_dict["op_select_implmode"])

    # 开启Allreduce和前后向并行执行
    custom_op.parameter_map["hcom_parallel"].b = config_dict["hcom_parallel"]

    if int(config_dict["graph_run_mode"]) == 1:
        print("MODE 1: train")
        custom_op.parameter_map["graph_run_mode"].i = 1

    elif int(config_dict["graph_run_mode"]) == 0:
        print("MODE 0: infer")
        custom_op.parameter_map["graph_run_mode"].i = 0

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    session = tf.Session(config=config, graph=graph)

    if force_as_default:
        # pylint: disable=protected-access
        session._default_session = session.as_default()
        session._default_session.enforce_nesting = False
        session._default_session.__enter__()  # pylint: disable=no-member

    return session


def init_uninitialized_vars(target_vars: List[tf.Variable] = None):
    """Initialize all tf.Variables that have not already been initialized.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tf.variables_initializer(tf.report_uninitialized_variables()).run()
    """
    assert_tf_initialized()
    if target_vars is None:
        target_vars = tf.global_variables()

    test_vars = []
    test_ops = []

    with tf.control_dependencies(None):  # ignore surrounding control_dependencies
        for var in target_vars:
            assert is_tf_expression(var)

            try:
                tf.get_default_graph().get_tensor_by_name(var.name.replace(":0", "/IsVariableInitialized:0"))
            except KeyError:
                # Op does not exist => variable may be uninitialized.
                test_vars.append(var)

                with absolute_name_scope(var.name.split(":")[0]):
                    test_ops.append(tf.is_variable_initialized(var))

    init_vars = [var for var, inited in zip(test_vars, run(test_ops)) if not inited]
    run([var.initializer for var in init_vars])


def set_vars(var_to_value_dict: dict) -> None:
    """Set the values of given tf.Variables.

    Equivalent to the following, but more efficient and does not bloat the tf graph:
    tfutil.run([tf.assign(var, value) for var, value in var_to_value_dict.items()]
    """
    assert_tf_initialized()
    ops = []
    feed_dict = {}

    for var, value in var_to_value_dict.items():
        assert is_tf_expression(var)

        try:
            setter = tf.get_default_graph().get_tensor_by_name(
                var.name.replace(":0", "/setter:0"))  # look for existing op
        except KeyError:
            with absolute_name_scope(var.name.split(":")[0]):
                with tf.control_dependencies(None):  # ignore surrounding control_dependencies
                    setter = tf.assign(var, tf.placeholder(var.dtype, var.shape, "new_value"),
                                       name="setter")  # create new setter

        ops.append(setter)
        feed_dict[setter.op.inputs[1]] = value

    run(ops, feed_dict)


def create_var_with_large_initial_value(initial_value: np.ndarray, *args, **kwargs):
    """Create tf.Variable with large initial value without bloating the tf graph."""
    assert_tf_initialized()
    assert isinstance(initial_value, np.ndarray)
    zeros = tf.zeros(initial_value.shape, initial_value.dtype)
    var = tf.Variable(zeros, *args, **kwargs)
    set_vars({var: initial_value})
    return var


def load_pb(model_file):
    """
    load fronzen graph
    :param model_file:
    :return:
    """
    with tf.gfile.GFile(model_file, "rb") as gf:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(gf.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


def broadcast_global_variables(root_rank, index):
    """Broadcasts all global variables from root rank to all other processes.
    Arguments:
    root_rank: rank of the process from which global variables will be broadcasted
    to all other processes.
    index: rank_id
    """
    op_list = []
    for var in tf.global_variables():
        # the input and out tensor of HCOMBroadcast interface are list
        if "float" in var.dtype.name:
            inputs = [var]
            outputs = hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)

            if outputs is not None:
                op_list.append(outputs[0].op)
                op_list.append(tf.assign(var, outputs[0]))

    return tf.group(op_list)

