import tensorflow as tf
import numpy as np

from .tokenize import tokenize

def make_translate(model, encoder_vocab, decoder_vocab, decoder_inverted_vocab, max_window_size = 10):
  def translate(sentence):
    sentence_tokens = [tokens + ['<stop>', '<pad>'] for tokens in [tokenize(sentence)]]
    tr_input = [list(map(lambda x: encoder_vocab[x], tokens)) for tokens in sentence_tokens][0]

    prediction = [[1]]
    i = 0

    while int(prediction[0][-1]) is not decoder_vocab['<stop>'] and i < max_window_size + 2:
      prediction_auto = model.predict([np.array([tr_input]), np.array(prediction)])
      prediction[0].append(tf.argmax(prediction_auto[0][i], axis = -1).numpy())
      i += 1

    print('Original: {}'.format(sentence))
    print('Traduction: {}'.format(' '.join(map(lambda x: decoder_inverted_vocab[x], prediction[0][1:-1]))))

  return translate
