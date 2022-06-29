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
"""
Encapsulates the worker which trains ChessModels using game data from recorded games from a file.
"""
from npu_bridge.npu_init import *
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from random import shuffle
import numpy as np
import time
# import precision_tool.tf_config as npu_tf_config
from chess_zero.agent.model_chess import ChessModel
from chess_zero.config import Config
from chess_zero.env.chess_env import canon_input_planes, is_black_turn, testeval
from chess_zero.lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from chess_zero.lib.model_helper import load_best_model_weight

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.data import Dataset
import tensorflow

logger = getLogger(__name__)


def start(config):
    """
    Helper method which just kicks off the optimization using the specified config
    :param Config config: config to use
    """
    return OptimizeWorker(config).start()


class OptimizeWorker:
    """
    Worker which optimizes a ChessModel by training it on game data

    Attributes:
        :ivar Config config: config for this worker
        :ivar ChessModel model: model to train
        :ivar dequeue,dequeue,dequeue dataset: tuple of dequeues where each dequeue contains game states,
            target policy network values (calculated based on visit stats
                for each state during the game), and target value network values (calculated based on
                    who actually won the game after that state)
        :ivar ProcessPoolExecutor executor: executor for running all of the training processes
    """
    def __init__(self, config):
        self.config = config
        self.model = None  # type: ChessModel
        self.dataset = deque(),deque(),deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)

    def start(self):
        """
        Load the next generation model from disk and start doing the training endlessly.
        """
        self.training()

    def training(self):
        """
        Does the actual training of the model, running it on game data. Endless.
        """
        if self.config.use_npu:
            sess_config = tf.ConfigProto()
            custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(16 * 1024 * 1024 * 1024))
            custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
            custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
            sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
            sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
            custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./fusion_switch.cfg")
            keras_sess = set_keras_session_npu_config(config = sess_config)
            K.set_session(keras_sess)
        else:
            config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=None,
                    allow_growth=None,
                )
            )
            keras_sess = tf.Session(config=config)
            K.set_session(keras_sess)
        
        self.model = self.load_model()
        self.compile_model()
        self.filenames = deque(get_game_data_filenames(self.config.resource))
        total_steps = self.config.trainer.start_total_steps

        epochs = 0
        if self.config.total_epochs > 0:
            epochs = self.config.total_epochs
        if self.config.total_epochs == -1:
            epochs = 10000

        while epochs > 0:
            self.fill_queue()
            steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            total_steps += steps
            # self.save_current_model()
            a, b, c = self.dataset
            while len(a) > self.config.trainer.dataset_size/2:
                a.popleft()
                b.popleft()
                c.popleft()
            epochs -= 1


    def train_epoch(self, epochs):
        """
        Runs some number of epochs of training
        :param int epochs: number of epochs
        :return: number of datapoints that were trained on in total
        """
        tc = self.config.trainer
        # state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        # tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        train_dataset, val_dataset, length = self.collect_all_loaded_data()
        startTime = time.time()
        self.model.model.fit(train_dataset,
                             epochs=epochs,
                             verbose=1,
                             # validation_data=val_dataset,
                             steps_per_epoch=int(length // tc.batch_size))
        endTime = time.time()
        steps = (int(length // tc.batch_size)) * epochs
        print(f"Model Train Performance: {((endTime - startTime) * 1000) / steps} ms/step")
        return steps

    def compile_model(self):
        """
        Compiles the model to use optimizer and loss function tuned for supervised learning
        """
        adam = tf.train.AdamOptimizer(learning_rate=1E-4, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
        #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        loss_scale_manager = FixedLossScaleManager(loss_scale=2 ** 10)
        opt = NPULossScaleOptimizer(adam, loss_scale_manager)
        losses = ['categorical_crossentropy', 'mean_squared_error'] # avoid overfit for supervised
        self.model.model.compile(optimizer=opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def save_current_model(self):
        """
        Saves the current model as the next generation model to the appropriate directory
        """
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def fill_queue(self):
        """
        Fills the self.dataset queues with data from the training dataset.
        """
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.popleft()
                logger.debug(f"loading data from {filename}")
                futures.append(executor.submit(load_data_from_file, filename))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                for x, y in zip(self.dataset, futures.popleft().result()):
                    x.extend(y)
                if len(self.filenames) > 0:
                    filename = self.filenames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file,filename))

    def collect_all_loaded_data(self):
        """
        :return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        tc = self.config.trainer
        state_ary,policy_ary,value_ary=self.dataset
        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        length = state_ary1.shape[0]
        split = int(length * 0.98)
        rng = np.random.RandomState(42)  # reproducible results with a fixed seed
        indices = np.arange(length)
        rng.shuffle(indices)
        state_ary1 = state_ary1[indices]
        policy_ary1 = policy_ary1[indices]
        value_ary1 = value_ary1[indices]
        state_train_ds = Dataset.from_tensor_slices(state_ary1[0:split, :, :, :])
        state_val_ds = Dataset.from_tensor_slices(state_ary1[split:length, :, :, :])
        pv_train_ds = Dataset.from_tensor_slices((policy_ary1[0:split, :], value_ary1[0:split]))
        pv_val_ds = Dataset.from_tensor_slices((policy_ary1[split:length, :], value_ary1[split:length]))
        train_dataset = Dataset.zip((state_train_ds, pv_train_ds)).batch(tc.batch_size,drop_remainder=True)
        val_dataset = Dataset.zip((state_val_ds, pv_val_ds)).batch(tc.batch_size,drop_remainder=True)
        return train_dataset, val_dataset, length*0.98

    def load_model(self):
        """
        Loads the next generation model from the appropriate directory. If not found, loads
        the best known model.
        """
        model = ChessModel(self.config)
        rc = self.config.resource

        dirs = get_next_generation_model_dirs(rc)
        if not dirs:
            logger.debug("loading best model")
            if not load_best_model_weight(model):
                raise RuntimeError("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            logger.debug("loading latest model")
            config_path = os.path.join(latest_dir, rc.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, rc.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)


def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:

        state_planes = canon_input_planes(state_fen)

        if is_black_turn(state_fen):
            policy = Config.flip_policy(policy)

        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number)/5 # reduces the noise of the opening... plz train faster
        sl_value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)

