import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def build(self, input_shape):
    super().build(input_shape)

  def call(self, x):
    input_shape = tf.shape(x)
    batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
    pos_input = tf.tile(tf.expand_dims(tf.keras.backend.arange(0, seq_len), axis = 0), [batch_size, 1])
    pos_input = tf.keras.backend.cast(pos_input, tf.float32)
    evens = tf.keras.backend.arange(0, output_dim // 2) * 2
    odds = tf.keras.backend.arange(0, output_dim // 2) * 2 + 1
    even_embedding = tf.sin(
      tf.keras.backend.dot(
        tf.expand_dims(pos_input, -1),
        tf.expand_dims(1.0 / tf.pow(
          10000.0,
          tf.cast(evens, dtype = tf.float32) / tf.cast(output_dim, dtype = tf.float32)
          ), 0)
        )
      )
    odd_embedding = tf.cos(
      tf.keras.backend.dot(
        tf.expand_dims(pos_input, -1),
        tf.expand_dims(1.0 / tf.pow(
          10000.0,
          tf.cast((odds - 1), dtype = tf.float32) / tf.cast(output_dim, dtype = tf.float32)
          ), 0)
        )
      )
    embedding = tf.stack([even_embedding, odd_embedding], axis = -1)
    output = tf.reshape(embedding, [-1, tf.shape(x)[1], output_dim])
    output += x

    return output
