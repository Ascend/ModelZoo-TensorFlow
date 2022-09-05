import tensorflow as tf

from .encoder_layer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
  def __init__(self, nb_encoder, embedding_size, dense_layer_size, nb_head, **kwargs):
    super().__init__(**kwargs)

    self.nb_encoder = nb_encoder
    self.embedding_size = embedding_size
    self.dense_layer_size = dense_layer_size
    self.nb_head = nb_head
    self.encoder_layers = []

  def get_config(self):
    config = super().get_config().copy()

    config.update({
      'nb_encoder': self.nb_encoder,
      'embedding_size': self.embedding_size,
      'dense_layer_size': self.dense_layer_size,
      'nb_head': self.nb_head
    })

    return config

  def build(self, input_shape):
    super().build(input_shape)

    for nb in range(self.nb_encoder):
      self.encoder_layers.append(
        EncoderLayer(self.embedding_size, self.dense_layer_size, self.nb_head)
      )

  def call(self, x):
    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x)
    return x
