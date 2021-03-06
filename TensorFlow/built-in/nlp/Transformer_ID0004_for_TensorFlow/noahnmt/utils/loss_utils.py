# coding=utf-8
# Copyright Huawei Noah's Ark Lab.
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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import tensorflow as tf

from noahnmt.utils import constant_utils
# from noahnmt.utils import align_utils
from noahnmt.layers import common_layers
from noahnmt.utils import graph_utils

# placeholders
PH_SYMBOLS = {"=SYM=", "=NUM=", "=DATE=", "=QUANT=", "TIME", "RANGE"}


def compute_nce_loss(logits, 
                     target, 
                     target_weight=None, 
                     label_smoothing=None):
  """Computes the loss for this model.

  Returns a tuple `(losses, loss)`, where `losses` are the per-batch
  losses and loss is a single scalar tensor to minimize.
  """
  # in case float16 training    
  if logits.dtype != tf.float32:
    logits = tf.cast(logits, dtype=tf.float32)

  shape_logits =  common_layers.shape_list(logits)

  target = tf.reshape(target, [-1])
  logits = tf.reshape(logits, [-1, shape_logits[-1]])
  
  if label_smoothing and label_smoothing > 0.:
    # using label smoothing
    # confidence = float(1. - label_smoothing)
    # confidence = tf.constant(confidence)
    # vocab_size = tf.shape(logits)[-1]
    # low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, dtype=tf.float32)
    # soft_labels = tf.one_hot(
    #     tf.cast(target, tf.int32),
    #     depth=vocab_size,
    #     on_value=confidence,
    #     off_value=low_confidence)
    # crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=logits, labels=tf.stop_gradient(soft_labels))

    confidence = float(1. - label_smoothing)
    vocab_size = tf.shape(logits)[-1]
    low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, dtype=tf.float32)
    soft_labels = tf.one_hot(
        tf.cast(target, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence,
        dtype=tf.float32)
    
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    
    crossent = tf.negative(tf.reduce_sum(log_probs * soft_labels, axis=-1))
  else:
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target, logits=logits)
  
  crossent = tf.reshape(crossent, shape_logits[:-1])

  graph_utils.add_dict_to_collection({"crossent": crossent}, "SAVE_TENSOR")
  
  if target_weight is not None:
    target_weight = tf.reshape(target_weight, [-1])
    crossent *= tf.cast(target_weight, crossent.dtype)

  return tf.reduce_sum(crossent), crossent


# def compute_alignment_loss(attention_scores, features, params):
#   """Guided alingment loss
#   """
#   weight = align_utils.weight_decay(
#       weight=params["guided_attention.weight"],
#       start_decay_step=params["guided_attention.start_decay_step"],
#       decay_steps=params["guided_attention.decay_steps"],
#       decay_factor=params["guided_attention.decay_rate"],
#       stop_decay_at=params["guided_attention.stop_decay_at"],
#   )
#   # batch x tgt_len x src_len
#   align_matrix = attention_scores
#   # batch x tgt_len x src_len
#   align_matrix_ref, align_mask = align_utils.create_dense_matrix(features)

#   if params["guided_attention.loss_type"] == "mse" \
#       or params["guided_attention.loss_type"] == "sqrt_mse":
#     align_loss_sum =  tf.reduce_sum(
#         ((align_matrix_ref - align_matrix) ** 2) * align_mask)
#     if params["guided_attention.loss_type"] == "sqrt_mse":
#       align_loss_sum = tf.sqrt(align_loss_sum)
#   elif params["guided_attention.loss_type"] == "ce":
#     align_matrix = tf.where(
#         align_matrix > 0,
#         align_matrix, 
#         tf.ones_like(align_matrix))
#     align_loss_sum =  - tf.reduce_sum(align_matrix_ref * tf.log(align_matrix) * align_mask)
#   else:
#     raise ValueError("Unknown loss type.")
#   return align_loss_sum * weight, None



# def _create_ph_mask(tokens):
#   mask = tf.fill(tf.shape(tokens), False)
#   for ph in PH_SYMBOLS:
#     ph_tensor = tf.constant(ph, tf.string)
#     mask = tf.logical_or(mask, tf.equal(tokens, ph_tensor))
#   return mask

# def placeholder_loss(features, attention_scores, word_level_loss):
#   """
#   calculate alignment loss on placeholder

#   Args:
#     features: dict with source_tokens and target_tokens
#     attention_scores: tensor with shape [batch, tgt_len, src_len]
#     word_level_loss: bool, use word-level or batch-level loss
  
#   Returns:
#     a float32 scalar
#   """
#   # shape [batch, length]
#   source_tokens = features["source_tokens"]
#   # shift because of <s> ... </s>
#   target_tokens = features["target_tokens"][:,1:]

  
#   # ph mask: 1.0 = ph, 0.0 = normal words
#   source_mask = tf.cast(
#       _create_ph_mask(source_tokens),
#       constant_utils.DT_FLOAT())
#   target_mask = tf.cast(
#       _create_ph_mask(target_tokens),
#       constant_utils.DT_FLOAT())    

#   # shape: [batch, tgt_len, src_len]
#   # = [batch, tgt_len, 1] x [batch, 1, src_len]
#   attention_mask = tf.matmul(
#       tf.expand_dims(target_mask, -1),
#       tf.expand_dims(source_mask, -1),
#       transpose_b=True)

#   # only keep attention scores on placeholder
#   att_scores = attention_scores * attention_mask
#   # shape: [batch, tgt_len]
#   loss_sum = tf.reduce_sum(att_scores, axis=2)
#   # take -log as loss
#   loss_sum = tf.where(target_mask>0, loss_sum, tf.ones_like(loss_sum))
#   loss_sum = tf.reduce_sum(-tf.log(loss_sum))
#   if word_level_loss:
#     loss_len = tf.reduce_sum(target_mask)
#   else:
#     # size of batch which contains placeholder
#     loss_len = tf.reduce_sum(
#         tf.cast(
#             tf.greater(
#                 tf.reduce_sum(target_mask, axis=1), 0),
#             constant_utils.DT_FLOAT()
#         )
#     )

#   return tf.cond(
#     loss_len > 0, 
#     lambda: loss_sum / loss_len,
#     lambda: tf.constant(0.0, dtype=constant_utils.DT_FLOAT()))
