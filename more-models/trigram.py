import numpy as np
from numpy.random import default_rng

np.set_printoptions(suppress=True)

def prep_names(file):
  words = []
  with open(file,'r') as f:
    for line in f.readlines():
      words.append('.' + line.replace("\n", "") + '.')
  return words

names = prep_names("datasets/names.txt")
chars = sorted(list(set(''.join(names))))

stoi = { s:i for i,s in enumerate(chars) }
#for k,v in stoi.items(): print(k, v)
assert stoi['.'] == 0
assert stoi['z'] == 26

itos = { i:s for s,i in stoi.items() }
#for k,v in itos.items(): print(k, v)
assert itos[0] == '.'
assert itos[26] == 'z'

# calculate bigram appearance counts
bigrams = np.zeros((27, 27))
getb = lambda a, b: bigrams[stoi[a], stoi[b]]
for name in names:
  for c1, c2 in zip(name, name[1:]):
    i1 = stoi[c1]
    i2 = stoi[c2]
    bigrams[i1, i2] += 1
assert getb('b', 'y') == ' '.join(names).count('by')

# calculate bigram normalized probability distributions
a = bigrams[1, 1]
sums = bigrams.sum(axis=1)
for i in range(27): 
  bigrams[i] = bigrams[i] / sums[i]
#print(bigrams[1, 1] * sums[1])
assert sum(bigrams[0]) == 1.0
assert bigrams[1, 1] * sums[1] == a

trigrams = np.zeros((27, 27, 27))
for name in names:
  for c1, c2, c3 in zip(name, name[1:], name[2:]):
    i1, i2, i3 = stoi[c1], stoi[c2], stoi[c3]
    trigrams[i1, i2, i3] += 1
gett = lambda a, b, c: trigrams[stoi[a], stoi[b], stoi[c]]
assert gett('a', 'r', 'k') == ' '.join(names).count('ark')

for x in range(27):
  for y in range(27):
    s = trigrams[x, y].sum()
    if s != 0.0:
      trigrams[x, y] = trigrams[x, y] / trigrams[x, y].sum()
assert trigrams[1, 1][0] * 556 == 40.0
  
def bigram_forward(n, seed=1337):
  g = default_rng(seed)
  for i in range(n):
    out = []
    ix = 0 # start at zero '.'
    while True:
      # predict next char
      probs = bigrams[ix]
      ix = g.multinomial(1, bigrams[ix]).argmax()
      s = itos[ix]
      out.append(itos[ix])
      # if prediction is end '.', break
      if ix == 0: break
    print(''.join(out))

def trigram_forward(n, seed=1337):
  g = default_rng(seed)
  for i in range(n):
    out = [] 
    c1 = 0
    c2 = g.multinomial(1, bigrams[0]).argmax()
    out.append(itos[c2])
    while True:
      probs = trigrams[c1, c2]
      c3 = g.multinomial(1, probs).argmax()
      s = itos[c3]
      out.append(s)
      if c3 == 0: break
      c1 = c2
      c2 = c3
    print(''.join(out))

if __name__ == "__main__":

  print("----- bigram generations -----")
  bigram_forward(10, 2458979)

  print("\n----- trigram generations -----")
  trigram_forward(10, 59048376)
