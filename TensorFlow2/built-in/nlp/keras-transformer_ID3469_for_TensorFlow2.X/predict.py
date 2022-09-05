import tensorflow as tf
import numpy as np

from dataset import get_dataset, prepare_dataset
from model import get_model
from utils.make_translate import make_translate


dataset = get_dataset("fr-en")

print("Dataset loaded. Length:", len(dataset), "lines")

train_dataset = dataset[0:100000]

print("Train data loaded. Length:", len(train_dataset), "lines")

(encoder_input,
decoder_input,
decoder_output,
encoder_vocab,
decoder_vocab,
encoder_inverted_vocab,
decoder_inverted_vocab) = prepare_dataset(
  train_dataset,
  shuffle = False,
  lowercase = True,
  max_window_size = 20
)

transformer_model = get_model(
  EMBEDDING_SIZE = 64,
  ENCODER_VOCAB_SIZE = len(encoder_vocab),
  DECODER_VOCAB_SIZE = len(decoder_vocab),
  ENCODER_LAYERS = 2,
  DECODER_LAYERS = 2,
  NUMBER_HEADS = 4,
  DENSE_LAYER_SIZE = 128
)

transformer_model.summary()

transformer_model.load_weights('./logs/transformer_ep-09_loss-0.14_acc-0.96.ckpt')

translate = make_translate(transformer_model, encoder_vocab, decoder_vocab, decoder_inverted_vocab)

translate("c'est une belle journée .")
translate("j'aime manger du gâteau .")
translate("c'est une bonne chose .")
translate("il faut faire à manger pour nourrir les gens .")
translate("tom a acheté un nouveau vélo .")

