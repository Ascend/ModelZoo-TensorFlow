# Copyright 2022 Huawei Technologies Co., Ltd
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
from npu_bridge.npu_init import *
from . import utils
import cv2
import numpy as np
import tensorflow as tf
import abc

ABC = abc.ABCMeta('ABC', (object,), {})

__all__ = [
    'PoseEstimatorInterface',
    'PoseEstimator'
]


class PoseEstimatorInterface(ABC):

    @abc.abstractmethod
    def initialise(self, args):
        pass

    @abc.abstractmethod
    def estimate(self, image):
        return

    @abc.abstractmethod
    def train(self, image, labels):
        return

    @abc.abstractmethod
    def close(self):
        pass


class PoseEstimator(PoseEstimatorInterface):

    def __init__(self, image_size, session_path, prob_model_path):
        """Initialising the graph in tensorflow.
        INPUT:
            image_size: Size of the image in the format (w x h x 3)"""

        self.session = None
        self.poseLifting = utils.Prob3dPose(prob_model_path)
        self.sess = -1
        self.orig_img_size = np.array(image_size)
        self.scale = utils.config.INPUT_SIZE / (self.orig_img_size[0] * 1.0)
        self.img_size = np.round(
            self.orig_img_size * self.scale).astype(np.int32)
        self.image_in = None
        self.heatmap_person_large = None
        self.pose_image_in = None
        self.pose_centermap_in = None
        self.pred_2d_pose = None
        self.likelihoods = None
        self.session_path = session_path

    def initialise(self):
        """Load saved model in the graph
        INPUT:
            sess_path: path to the dir containing the tensorflow saved session
        OUTPUT:
            sess: tensorflow session"""
        # initialize graph structrue
        tf.reset_default_graph()

        with tf.variable_scope('CPM'):
            # placeholders for person network
            self.image_in = tf.placeholder(
                tf.float32, [None, utils.config.INPUT_SIZE, self.img_size[1], 3])
            self.label_in = tf.placeholder(
                tf.float32, [None, utils.config.INPUT_SIZE, self.img_size[1], 1])

            heatmap_person = utils.inference_person(self.image_in)

            self.heatmap_person_large = tf.image.resize_images(
                heatmap_person, [utils.config.INPUT_SIZE, self.img_size[1]])

            # placeholders for pose network
            self.pose_image_in = tf.placeholder(
                tf.float32,
                [utils.config.BATCH_SIZE, utils.config.INPUT_SIZE, utils.config.INPUT_SIZE, 3])

            self.pose_centermap_in = tf.placeholder(
                tf.float32,
                [utils.config.BATCH_SIZE, utils.config.INPUT_SIZE, utils.config.INPUT_SIZE, 1])

            self.pred_2d_pose, self.likelihoods = utils.inference_pose(
                self.pose_image_in, self.pose_centermap_in,
                utils.config.INPUT_SIZE)

        # set up loss and optimizer
        self.loss = tf.reduce_mean(tf.abs(self.heatmap_person_large - self.label_in))
        self.optimizer = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=0.0000001)).minimize(self.loss)

        # load pretraining model
        sess = tf.Session(config=npu_config_proto())
        sess.run(tf.global_variables_initializer())
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if v.name.split('/')[-1][:4] != 'Adam' and v.name[:4] != 'beta']
        self.saver = tf.train.Saver(variables_to_resotre)
        self.saver.restore(sess, self.session_path)
        self.session = sess

    def train(self, image, labels):
        # input model,back propagation and then output loss
        b_image = np.array(image / 255.0 - 0.5, dtype=np.float32)
        labels = labels[:, :, :, np.newaxis]

        # self.session.run(self.optimizer, {self.image_in: b_image, self.label_in: labels})
        _, loss, heatmap_pred = self.session.run([self.optimizer, self.loss, self.heatmap_person_large],
                                                 feed_dict={self.image_in: b_image, self.label_in: labels})
        return loss, heatmap_pred

    def estimate(self, image, lifting=False):
        """
        Estimate 2d and 3d poses on the image.
        INPUT:
            image: RGB image in the format (w x h x 3)
            sess: tensorflow session
        OUTPUT:
            pose_2d: 2D pose for each of the people in the image in the format
            (num_ppl x num_joints x 2)
            visibility: vector containing a bool
            value for each joint representing the visibility of the joint in
            the image (could be due to occlusions or the joint is not in the
            image)
            pose_3d: 3D pose for each of the people in the image in the
            format (num_ppl x 3 x num_joints)
            hmap_person: heatmap
        """
        # test model
        sess = self.session

        image = cv2.resize(image, (0, 0), fx=self.scale,
                           fy=self.scale, interpolation=cv2.INTER_CUBIC)
        b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)

        hmap_person_viz = sess.run(self.heatmap_person_large, {
            self.image_in: b_image})
        hmap_person = np.squeeze(hmap_person_viz)

        centers = utils.detect_objects_heatmap(hmap_person)
        b_pose_image, b_pose_cmap = utils.prepare_input_posenet(
            b_image[0], centers,
            [utils.config.INPUT_SIZE, image.shape[1]],
            [utils.config.INPUT_SIZE, utils.config.INPUT_SIZE],
            batch_size=utils.config.BATCH_SIZE)

        feed_dict = {
            self.pose_image_in: b_pose_image,
            self.pose_centermap_in: b_pose_cmap
        }

        # Estimate 2D poses
        pred_2d_pose, pred_likelihood = sess.run([self.pred_2d_pose,
                                                  self.likelihoods],
                                                 feed_dict)

        estimated_2d_pose, visibility = utils.detect_parts_from_likelihoods(pred_2d_pose,
                                                                            centers,
                                                                            pred_likelihood)

        pose_2d = np.round(estimated_2d_pose / self.scale).astype(np.int32)
        
        # Estimate 3D poses
        if lifting:
            transformed_pose2d, weights = self.poseLifting.transform_joints(
                estimated_2d_pose.copy(), visibility)
            pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
            return pose_2d, visibility, pose_3d

        return pose_2d, hmap_person
    def close(self):
        self.session.close()
