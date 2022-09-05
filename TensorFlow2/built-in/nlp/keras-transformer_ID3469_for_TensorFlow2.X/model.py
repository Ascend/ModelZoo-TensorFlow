import tensorflow as tf

from models.word_embedding import WordEmbedding
from models.positional_embedding import PositionalEmbedding
from models.decoder import Decoder
from models.encoder import Encoder

def get_model(
    EMBEDDING_SIZE = 64,
    DENSE_LAYER_SIZE = 128,
    ENCODER_VOCAB_SIZE = 12,
    DECODER_VOCAB_SIZE = 12,
    ENCODER_LAYERS = 1,
    DECODER_LAYERS = 1,
    NUMBER_HEADS = 1,
    MAX_WIN_SIZE = 20,
    BATCH_SIZE = 32
  ):

  encoder_layer_input = tf.keras.Input(shape = (MAX_WIN_SIZE,), batch_size=BATCH_SIZE, name = "Encoder-Input")
  decoder_layer_input = tf.keras.Input(shape = (MAX_WIN_SIZE,), batch_size=BATCH_SIZE, name = "Decoder-Input")

  encoder_embedding = WordEmbedding(ENCODER_VOCAB_SIZE, EMBEDDING_SIZE, name = "Encoder-Word-Embedding")(encoder_layer_input)
  decoder_embedding = WordEmbedding(DECODER_VOCAB_SIZE, EMBEDDING_SIZE, name = "Decoder-Word-Embedding")(decoder_layer_input)

  encoder_embedding = PositionalEmbedding(name = "Encoder-Positional-Embedding")(encoder_embedding)
  decoder_embedding = PositionalEmbedding(name = "Decoder-Positional-Embedding")(decoder_embedding)

  encoder_output = Encoder(ENCODER_LAYERS, EMBEDDING_SIZE, DENSE_LAYER_SIZE, NUMBER_HEADS, name = "Encoder")(encoder_embedding)
  decoder_output = Decoder(DECODER_LAYERS, EMBEDDING_SIZE, DENSE_LAYER_SIZE, NUMBER_HEADS, name = "Decoder")((decoder_embedding, encoder_output))

  output_predictions = tf.keras.layers.Dense(DECODER_VOCAB_SIZE, activation = "softmax", name = "Decoder-Output")(decoder_output)

  model = tf.keras.Model([encoder_layer_input, decoder_layer_input], output_predictions, name = "Transformer-Model")

  return model
