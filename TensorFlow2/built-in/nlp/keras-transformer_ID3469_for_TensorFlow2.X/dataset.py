from utils.shuffle_list import shuffle_list
from utils.tokenize import tokenize
from utils.get_vocab import get_vocab
from utils.encode_with_vocab import encode_with_vocab
import os

def get_dataset(name, path):
  if (name == "fr-en"):
    f = open(os.path.join(path, "fr-en-train.txt"), "r")
    file = f.read()
    return file.split("\n")
  else:
    raise SystemError("Dataset not found")

def prepare_dataset(dataset, shuffle, lowercase, max_window_size):
  encoder_input, decoder_input, decoder_output = [], [], []
  encoder_vocab, decoder_vocab, encoder_inverted_vocab, decoder_inverted_vocab = {}, {}, {}, {}

  if shuffle:
    dataset = shuffle_list(dataset)

  for line in dataset:
    if lowercase:
      line = line.lower()
    en, fr, credits = line.split("\t")

    encoder_input.append(tokenize(fr))
    decoder_input.append(tokenize(en))

  decoder_output = [tokens + ["<stop>"] for tokens in decoder_input]
  encoder_input = [["<start>"] + tokens + ["<stop>"] for tokens in encoder_input]
  decoder_input = [["<start>"] + tokens + ["<stop>"] for tokens in decoder_input]

  source_max_len = max_window_size
  target_max_len = max_window_size
  if (max(map(len, encoder_input)) > max_window_size or max(map(len, decoder_input)) > max_window_size):
    raise SystemError("Maximum window size is too small", max(map(len, encoder_input)), max(map(len, decoder_input)))

  encoder_input = [tokens + ["<pad>"] * (source_max_len - len(tokens)) for tokens in encoder_input]
  decoder_input = [tokens + ["<pad>"] * (target_max_len - len(tokens)) for tokens in decoder_input]
  decoder_output = [tokens + ["<pad>"] * (target_max_len - len(tokens)) for tokens in decoder_output]

  encoder_vocab = get_vocab(encoder_input)
  decoder_vocab = get_vocab(decoder_input)

  encoder_inverted_vocab = { v: k for k, v in encoder_vocab.items() }
  decoder_inverted_vocab = { v: k for k, v in decoder_vocab.items() }

  encoder_input = encode_with_vocab(encoder_input, encoder_vocab)
  decoder_input = encode_with_vocab(decoder_input, decoder_vocab)
  decoder_output = encode_with_vocab(decoder_output, decoder_vocab)
  decoder_output = [[[token] for token in tokens] for tokens in decoder_output]

  return (encoder_input,
    decoder_input,
    decoder_output,
    encoder_vocab,
    decoder_vocab,
    encoder_inverted_vocab,
    decoder_inverted_vocab)
