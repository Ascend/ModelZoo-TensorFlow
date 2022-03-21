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

"""Implementation of the Frechet Inception Distance.

Implemented as a wrapper around the tf.contrib.gan library. The details can be
found in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
Equilibrium", Heusel et al. [https://arxiv.org/abs/1706.08500].
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from npu_bridge.estimator import npu_ops
from npu_bridge.estimator.npu import npu_scope
from absl import logging
from compare_gan.asess_util import config
from compare_gan.metrics import eval_task

import tensorflow as tf

import tensorflow_gan as tfgan


# Special value returned when FID code returned exception.
FID_CODE_FAILED = 4242.0


class FIDScoreTask(eval_task.EvalTask):
  """Evaluation task for the FID score."""

  _LABEL = "fid_score"

  def run_after_session(self, fake_dset, real_dset):
    logging.info("Calculating FID.")
    with tf.Graph().as_default():
      fake_activations = tf.convert_to_tensor(fake_dset.activations)
      real_activations = tf.convert_to_tensor(real_dset.activations)
      fid = frechet_classifier_distance_from_activations(
          real_activations=real_activations,
          generated_activations=fake_activations)
      with self._create_session() as sess:
        fid = sess.run(fid)
      logging.info("Frechet Inception Distance: %.3f.", fid)
      return {self._LABEL: fid}


def compute_fid_from_activations(fake_activations, real_activations):
  """Returns the FID based on activations.

  Args:
    fake_activations: NumPy array with fake activations.
    real_activations: NumPy array with real activations.
  Returns:
    A float, the Frechet Inception Distance.
  """
  logging.info("Computing FID score.")
  assert fake_activations.shape == real_activations.shape
  with tf.Session(graph=tf.Graph(),config=config) as sess:
    fake_activations = tf.convert_to_tensor(fake_activations)
    real_activations = tf.convert_to_tensor(real_activations)
    fid = frechet_classifier_distance_from_activations(
      real_activations=real_activations,
      generated_activations=fake_activations)
    return sess.run(fid)

def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
  """Classifier distance for evaluating a generative model.

  This methods computes the Frechet classifier distance from activations of
  real images and generated images. This can be used independently of the
  frechet_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like precompute all of the
  activations before computing the classifier distance.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

                |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.

  """
  real_activations.shape.assert_has_rank(2)
  generated_activations.shape.assert_has_rank(2)

  activations_dtype = real_activations.dtype
  if activations_dtype != tf.float32:
    real_activations = tf.cast(real_activations, tf.float32)
    generated_activations = tf.cast(generated_activations, tf.float32)

  # Compute mean and covariance matrices of activations.
  m = tf.reduce_mean(real_activations, 0)
  m_w = tf.reduce_mean(generated_activations, 0)
  num_examples_real = tf.cast(tf.shape(real_activations)[0], tf.float32)
  num_examples_generated = tf.cast(
      tf.shape(generated_activations)[0], tf.float32)

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  real_centered = real_activations - m
  sigma = tf.matmul(
      real_centered, real_centered, transpose_a=True) / (
          num_examples_real - 1)

  gen_centered = generated_activations - m_w
  sigma_w = tf.matmul(
      gen_centered, gen_centered, transpose_a=True) / (
          num_examples_generated - 1)

  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = tf.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = tf.reduce_sum(tf.squared_difference(
      m, m_w))  # Equivalent to L2 but more stable.
  fid = trace + mean
  if activations_dtype != tf.float32:
    fid = tf.cast(fid, activations_dtype)

  return fid


def trace_sqrt_product(sigma, sigma_v):

  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

  return tf.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))

def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  # Unlike numpy, tensorflow's return order is (s, u, v)
  s, u, v = tf.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
  # Note that the v returned by Tensorflow is v = V
  # (when referencing the equation A = U S V^T)
  # This is unlike Numpy which returns v = V^T
  return tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)
