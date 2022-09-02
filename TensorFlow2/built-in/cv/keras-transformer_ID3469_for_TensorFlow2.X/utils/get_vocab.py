def get_vocab(sequences):
  token_id_map = {
      "<pad>": 0,
      "<start>": 1,
      "<stop>": 2
  }

  for sequence in sequences:
    for word in sequence:
      if word not in token_id_map:
        token_id_map[word] = len(token_id_map)

  return token_id_map
