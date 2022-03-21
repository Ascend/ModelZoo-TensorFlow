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


"""Attention file for location based attention (compatible with tensorflow attention wrapper)"""
from npu_bridge.npu_init import *

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope

from synthesizer.hparams import hparams
#From https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py
def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer):
	"""Computes the attention and alignments for a given attention_mechanism."""
	
	#TODO: changed following
	alignments, next_attention_state, processed_query = attention_mechanism(
		cell_output, state=attention_state)

	# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# Context is the inner product of alignments and values along the
	# memory time dimension.
	# alignments shape is
	#   [batch_size, 1, memory_time]
	# attention_mechanism.values shape is
	#   [batch_size, memory_time, memory_size]
	# the batched matmul is over memory_time, so the output shape is
	#   [batch_size, 1, memory_size].
	# we then squeeze out the singleton dim.
	# print("expanded_alignments {}". format(expanded_alignments.shape))
	# print("expanded_alignments {}". format(attention_mechanism.values.shape))
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state, processed_query


def _location_sensitive_score(W_query, W_fil, W_keys):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################

	Args:
		W_query: Tensor, shape "[batch_size, 1, attention_dim]" to compare to location features.
		W_location: processed previous alignments into location features, shape "[batch_size, max_time, attention_dim]"
		W_keys: Tensor, shape "[batch_size, max_time, attention_dim]", typically the encoder outputs.
	Returns:
		A "[batch_size, max_time]" attention score (energy)
	"""
	# Get the number of hidden units from the trailing dimension of keys
	dtype = W_query.dtype
	num_units = 128 #W_keys.shape[-1].value #or array_ops.shape(W_keys)[-1]
	
	#TODO:
	#num_units = hparams.attention_dim

	#print("dtype : ", dtype)

	#print ("num units prev: ", num_units, type(num_units)) #type(W_keys.shape[-1].value) or type(array_ops.shape(W_keys)[-1]))

	v_a = tf.get_variable(
		"attention_variable_projection", shape=[1,1,num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		"attention_bias", shape=[1,1,num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	# print ("v_a, v_b shape : ", v_a.shape, b_a.shape)

	# print ("location sensitive param W_query : ", W_query.shape)

	# print ("location sensitive param W_fil : ", W_fil.shape)

	# print ("location sensitive param W_keys : ", W_keys.shape)

	# print ("sum check : ", tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2]))
	W_query_tiled = tf.tile(W_query, [1,hparams.T,1])
	b_a_tiled = tf.tile(b_a,[hparams.tacotron_batch_size,hparams.T,1])
	v_a_tiled = tf.tile(v_a,[hparams.tacotron_batch_size,hparams.T,1])
	return tf.reduce_sum(v_a_tiled * tf.tanh(W_keys + W_query_tiled + W_fil + b_a_tiled), [2])
	#return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])

def _smoothing_normalization(e):
	"""Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)

class LocationSensitiveAttention(BahdanauAttention):

	"""Impelements Bahdanau-style (cumulative) scoring function.
	Usually referred to as "hybrid" attention (content-based + location-based)
	Extends the additive attention described in:
	"D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
	to use previous alignments as additional location features.

	This attention is described in:
	J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
	"""

	def __init__(self,
				 num_units,
				 memory,
				 hparams,
				 mask_encoder=True,
				 memory_sequence_length=None,
				 smoothing=False,
				 cumulate_weights=True,
				 name="LocationSensitiveAttention",
				 probability_fn = None):
		"""Construct the Attention mechanism.
		Args:
			num_units: The depth of the query mechanism.
			memory: The memory to query; usually the output of an RNN encoder.  This
				tensor should be shaped `[batch_size, max_time, ...]`.
			mask_encoder (optional): Boolean, whether to mask encoder paddings.
			memory_sequence_length (optional): Sequence lengths for the batch entries
				in memory.  If provided, the memory tensor rows are masked with zeros
				for values past the respective sequence lengths. Only relevant if mask_encoder = True.
			smoothing (optional): Boolean. Determines which normalization function to use.
				Default normalization function (probablity_fn) is softmax. If smoothing is
				enabled, we replace softmax with:
						a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
				Introduced in:
					J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
				  gio, “Attention-based models for speech recognition,” in Ad-
				  vances in Neural Information Processing Systems, 2015, pp.
				  577–585.
				This is mainly used if the model wants to attend to multiple input parts
				at the same decoding step. We probably won"t be using it since multiple sound
				frames may depend on the same character/phone, probably not the way around.
				Note:
					We still keep it implemented in case we want to test it. They used it in the
					paper in the context of speech recognition, where one phoneme may depend on
					multiple subsequent sound frames.
			name: Name to use when creating ops.
		"""
		#Create normalization function
		#Setting it to None defaults in using softmax
		normalization_function = _smoothing_normalization if (smoothing == True) else None
		memory_length = memory_sequence_length if (mask_encoder==True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=memory_length,
				probability_fn=normalization_function,
				name=name)

		self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
			kernel_size=hparams.attention_kernel, padding="same", use_bias=True,
			bias_initializer=tf.zeros_initializer(), name="location_features_convolution")
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,
			dtype=tf.float32, name="location_features_layer")
		self._cumulate = cumulate_weights

		# if probability_fn is None:
		# 	self._probability_fn = tf.nn.softmax
		#self.processed_query = None

	def __call__(self, query, state):
		"""Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory"s `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# TODO:	
			
			# -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)
			

			# processed_location_features shape [batch_size, max_time, attention dimension]
			# [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			

			# location features [batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			
			# Projected location features [batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)

			#self.processed_location_features = processed_location_features
		
			# energy shape [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)
			
			# self.processed_query = energy
			# print ("inside call ", self.processed_query)

		# alignments shape = energy shape = [batch_size, max_time]
		alignments = tf.nn.softmax(energy) #self._probability_fn(energy) #,previous_alignments) #tf.nn.softmax(energy) #self._probability_fn(energy, previous_alignments) #check with GPU

		# self.processed_query = alignments
		# print ("inside call ", self.processed_query)

		# Cumulate alignments
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments

		self.processed_query = next_state
		print ("inside call ", self.processed_query)

		return alignments, next_state, self.processed_query

