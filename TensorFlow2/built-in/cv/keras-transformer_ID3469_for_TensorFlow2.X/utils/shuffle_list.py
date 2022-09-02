import random

random.seed(0)

def shuffle_list(list):
  shuffled = list.copy()
  random.shuffle(shuffled)

  return shuffled
