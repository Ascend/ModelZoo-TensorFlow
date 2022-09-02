import tensorflow as tf

from .attentive_pooling import AttentionWithContext
from .self_attention.encoder import EncoderLayer
from .self_attention.positional_encoding import PositionalEncoding
from .sensor_attention import SensorAttention

from rich.console import Console
CONSOLE = Console()


def create_model(n_timesteps, n_features, n_outputs, _dff=512, d_model=128, nh=4, dropout_rate=0.2, use_pe=True):
    """ The idea of this paper is that sensor's data samples are equivalent
        to words and windows (time windows) are analogous to sentence.
        Hence the use of attention & transformers for the paper.
        The purpose has been to build an attention based end-to-end
        system where attention is utilized in different ways to create
        effective feature representation for sensor data.

        This is a self-attention based model. It utilizes sensor modality
        attention, self-attention blocks and global temporal attention.

        The input is a time-window of sensor values. First it applies sensor
        modality to get a weighted representation of the sensor values
        according to their attention score. This learned attention score
        represents the contribution of each of the sensor modalities in the
        feature representation used by subsequent layers.

        Afterwards, the weighted sensor values are converted to `d` size
        vectors using 1-D convolution over single time-steps

        Optionally, positional information of the samples is encoded by adding
        values based on sine and cosine functions to the obtained `d` size
        vectors. This enables the model to take the temporal order of samples
        into account. After that, the representation is scaled by
        square_root(d) and is passed to the self-attention blocks.

        These self-attention blocks use dot product-based attention score to
        transform the feature values for each timestamp.

        Finally, the representation generated from the self attention blocks
        is used by the global temporal attention layer.
        This layer learns parameters to set varying attention across the
        temporal dimension to generate the final representation which is used
        by the final fully connected and softmax layers."""

    CONSOLE.print('=====     Hyperparameters     =====', style='green')
    CONSOLE.print(f'Time steps: {n_timesteps}; Number Features: {n_features}'
                  f' Number Outputs: {n_outputs}; D Model: {d_model}'
                  f' Number Heads: {nh}; Dropout: {dropout_rate}'
                  f' Positional Encoding: {use_pe}', style='green')

    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features,))

    # si, _ = SensorAttention(n_filters=256, kernel_size=5, dilation_rate=2)(inputs)

    x = tf.keras.layers.Conv1D(d_model, 1, activation='relu')(inputs)

    if use_pe:
        x = PositionalEncoding(n_timesteps, d_model, dropout_rate)(x)

    # add self-attention layers
    x = EncoderLayer(
        d_model=d_model, num_heads=nh,
        dff=_dff, rate=dropout_rate)(x)
    x = EncoderLayer(
        d_model=d_model, num_heads=nh,
        dff=_dff, rate=dropout_rate)(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = AttentionWithContext()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(n_outputs * 4, activation='relu')(x)
    # x = tf.keras.layers.Dropout(dropout_rate)(x)
    # x = tf.keras.layers.Dense(128, activation='relu') (x)

    predictions = tf.keras.layers.Dense(n_outputs, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model
