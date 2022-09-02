from nltk import RegexpTokenizer

toknizer = RegexpTokenizer(r'''\w+'|[\w-]+|[^\w\s]''')

def tokenize(sentence):
  return toknizer.tokenize(sentence)
