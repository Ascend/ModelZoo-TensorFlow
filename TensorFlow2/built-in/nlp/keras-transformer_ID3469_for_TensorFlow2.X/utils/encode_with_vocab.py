def encode_with_vocab(sequences, vocab):
  encoded_sequences = []
  encoded_sequence = []

  for sequence in sequences:
    for word in sequence:
      encoded_sequence.append(vocab[word])
    encoded_sequences.append(encoded_sequence)
    encoded_sequence = []

  return encoded_sequences
