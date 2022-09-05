import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, embedding_size, nb_head, **kwargs):
    super().__init__(**kwargs)

    if not embedding_size % nb_head == 0:
      raise SystemError("Embedding_size should be divisible by number of heads")

    self.embedding_size = embedding_size
    self.nb_head = nb_head
    self.head_dim = embedding_size // nb_head

  def get_config(self):
    config = super().get_config().copy()

    config.update({
      'embedding_size': self.embedding_size,
      'nb_head': self.nb_head
    })

    return config

  def build(self, input_shape):
    super().build(input_shape)

    self.query_layer = tf.keras.layers.Dense(self.embedding_size)
    self.value_layer = tf.keras.layers.Dense(self.embedding_size)
    self.key_layer = tf.keras.layers.Dense(self.embedding_size)
    self.out_proj = tf.keras.layers.Dense(self.embedding_size)

  def call(self, x, mask = False):
    Q_input, K_input, V_input = x

    Q = self.query_layer(Q_input)
    K = self.key_layer(K_input)
    V = self.value_layer(V_input)

    if self.nb_head > 1:
      batch_size = tf.shape(Q)[0]
      Q_seq_len = tf.shape(Q)[1]
      K_seq_len = tf.shape(K)[1]
      V_seq_len = tf.shape(V)[1]

      Q = tf.reshape(Q, [batch_size, Q_seq_len, self.nb_head, self.head_dim])
      K = tf.reshape(K, [batch_size, K_seq_len, self.nb_head, self.head_dim])
      V = tf.reshape(V, [batch_size, V_seq_len, self.nb_head, self.head_dim])

      Q = tf.transpose(Q, [0, 2, 1, 3])
      K = tf.transpose(K, [0, 2, 1, 3])
      V = tf.transpose(V, [0, 2, 1, 3])

      Q = tf.reshape(Q, [batch_size * self.nb_head, Q_seq_len, self.head_dim])
      K = tf.reshape(K, [batch_size * self.nb_head, K_seq_len, self.head_dim])
      V = tf.reshape(V, [batch_size * self.nb_head, V_seq_len, self.head_dim])

    dot_product = tf.matmul(Q, K, transpose_b = True)
    scaled_dot_product = dot_product / tf.math.sqrt(float(self.embedding_size))

    if mask:
      diag_vals = tf.ones_like(scaled_dot_product[0, :, :])
      tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
      future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(scaled_dot_product)[0], 1, 1])
      padding_num = -float("Inf")
      paddings = tf.ones_like(future_masks) * padding_num

      scaled_dot_product = tf.where(tf.equal(future_masks, 0), paddings, scaled_dot_product)

    softmax_product = tf.nn.softmax(scaled_dot_product, axis = -1)
    attention = tf.matmul(softmax_product, V)

    if self.nb_head > 1:
      attention = tf.reshape(
        attention, [batch_size, self.nb_head, Q_seq_len, self.head_dim]
      )

      attention = tf.transpose(attention, [0, 2, 1, 3])

      attention = tf.reshape(
        attention, [batch_size, Q_seq_len, self.nb_head * self.head_dim]
      )

    out_attention = self.out_proj(attention)

    return out_attention
