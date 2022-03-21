# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
tf = None
from npu_bridge.npu_init import *
import warnings

def _import_tf():
    """ Tries to import tensorflow.
    """
    global tf
    if tf is None:
        import tensorflow as tf

def _get_session(session):
    """ Common utility to get the session for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.

    session : tf.compat.v1.Session

        An optional existing session.
    """
    _import_tf()
    # if we are not given a session find a default session
    if session is None:
        try:
            session = tf.compat.v1.keras.backend.get_session()
        except:
            session = tf.keras.backend.get_session()
    return tf.get_default_session() if session is None else session

def _get_graph(explainer):
    """ Common utility to get the graph for the tensorflow-based explainer.

    Parameters
    ----------
    explainer : Explainer

        One of the tensorflow-based explainers.
    """
    _import_tf()
    if True:
        return explainer.session.graph
    else:
        from tensorflow.python.keras import backend
        graph = backend.get_graph()
        return graph

def _get_model_inputs(model):
    """ Common utility to determine the model inputs.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.
    """
    _import_tf()
    if str(type(model)).endswith("keras.engine.sequential.Sequential'>") or \
        str(type(model)).endswith("keras.models.Sequential'>") or \
        str(type(model)).endswith("keras.engine.training.Model'>") or \
        isinstance(model, tf.keras.Model):
        return model.inputs
    elif str(type(model)).endswith("tuple'>"):
        return model[0]
    else:
        assert False, str(type(model)) + " is not currently a supported model type!"

def _get_model_output(model):
    """ Common utility to determine the model output.

    Parameters
    ----------
    model : Tensorflow Keras model or tuple

        The tensorflow model or tuple.
    """
    _import_tf()
    if str(type(model)).endswith("keras.engine.sequential.Sequential'>") or \
        str(type(model)).endswith("keras.models.Sequential'>") or \
        str(type(model)).endswith("keras.engine.training.Model'>") or \
        isinstance(model, tf.keras.Model):
        if len(model.layers[-1]._inbound_nodes) == 0:
            if len(model.outputs) > 1:
                warnings.warn("Only one model output supported.")
            return model.outputs[0]
        else:
            return model.layers[-1].output
    elif str(type(model)).endswith("tuple'>"):
        return model[1]
    else:
        assert False, str(type(model)) + " is not currently a supported model type!"

