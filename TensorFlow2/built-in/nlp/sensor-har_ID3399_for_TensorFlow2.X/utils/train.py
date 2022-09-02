import os
import sys
import time
import warnings

import tensorflow as tf

from model.har_model import create_model

tf.keras.backend.clear_session()
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../")

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        self.start_time = None
    
    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
        self.epoch_start = time.time()
    
    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size
            print(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d'%(elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps),flush=True)
            self.last_log_step = self.global_steps
            self.start_time = None
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

def train_model(dataset: str, model_config, train_x, train_y, val_x, val_y, epochs, batch_size=128, log_steps=232):
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    model = create_model(
                        n_timesteps,
                        n_features,
                        n_outputs,
                        d_model=model_config[dataset]['d_model'],
                        nh=model_config[dataset]['n_head'],
                        dropout_rate=model_config[dataset]['dropout'])

    model.compile(**model_config['training'])
    model.summary()

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                          factor=0.1,
                                                          patience=10, # ? modify
                                                          verbose=1,
                                                          min_delta=1e-4,
                                                          mode='min')

    model.fit(train_x, train_y,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              validation_data=(val_x, val_y),
              callbacks=[reduce_lr_loss, TimeHistory(batch_size, log_steps)])

    print(f'Saving trained model for {dataset}')
    model.save_weights(filepath="checkpoint/tf_model", save_format="tf")
