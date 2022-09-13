import tensorflow as tf

# https://keras.io/guides/making_new_layers_and_models_via_subclassing/


class PositionalEncoding(tf.keras.layers.Layer):
    """ A layer used in Transformer architecture that gives the order context
        to the non-recurrent architecture of multi-head attention. In other
        words this layer seems to be crucial for ordering time series data.

        Basically, when recurrent networks are fed with sequence input, the
        sequential order (ordering of time-steps) is implicitly defined by the
        input. However, the Multi-Head Attention layer in the Transformer is a
        feed-forward layer that reads a whole sequence at once. As the
        attention is computed on each data point (time-step) independently,
        the context of ordering between data points is simply lost and the
        attention is invariant to the sequence order. (same holds for CNN)
        Hence we need this layer to take care of the sequential nature of
        time-series data

        https://medium.com/@j.ali.hab/on-positional-encodings-in-the-attention-mechanism-ee81e6076b62
    """

    def __init__(self, position, d_model,
                 dropout_rate=0.2, include_dropout=True):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
        self.include_dropout = include_dropout
        if include_dropout:
            self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)


    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles


    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)


    def call(self, inputs):
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs = inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        if self.include_dropout:
            inputs = self.dropout(inputs)
        return inputs
