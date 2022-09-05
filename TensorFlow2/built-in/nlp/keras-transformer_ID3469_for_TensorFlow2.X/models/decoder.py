import tensorflow as tf

from .decoder_layer import DecoderLayer

class Decoder(tf.keras.layers.Layer):
  def __init__(self, nb_decoder, embedding_size, dense_layer_size, nb_head = 1, **kwargs):
    super().__init__(**kwargs)

    self.nb_decoder = nb_decoder
    self.embedding_size = embedding_size
    self.dense_layer_size = dense_layer_size
    self.decoder_layers = []
    self.nb_head = nb_head

  def get_config(self):
    config = super().get_config().copy()

    config.update({
      'nb_decoder': self.nb_decoder,
      'embedding_size': self.embedding_size,
      'dense_layer_size': self.dense_layer_size,
      'nb_head': self.nb_head
    })

    return config

  def build(self, input_shape):
    super().build(input_shape)

    for nb in range(self.nb_decoder):
      self.decoder_layers.append(
        DecoderLayer(self.embedding_size, self.dense_layer_size, self.nb_head)
      )

  def call(self, x):
    output_embedding, encoder_output = x

    decoder_output = output_embedding

    for decoder_layer in self.decoder_layers:
      decoder_output = decoder_layer((decoder_output, encoder_output))

    return decoder_output
