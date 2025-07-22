import torch 
import matplotlib.pyplot as plt
words = open('names.txt', 'r').read().splitlines()
# y = words[:10]
# print(y)
# print(len(words) ,min(len(w) for w in words),max(len(w) for w in words))
# for w in words[:3]:
#     for ch1, ch2 in zip(w, w[1:]):
#         print(ch1, ch2)
# # # concepts of zip 
# # for w in words[:1]:
# #     for ch1 in zip(w):
# #         print(ch1)
# # print(w, w[:1], w[1:])
d = {}##dictonary                                                          
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs , chs[1:]):
     bigram = (ch1, ch2)
     d[bigram] = d.get(bigram, 0) + 1
    
    
    #print(d)
# for w in words:
#     pass
#     print(w)
chars = set()

for w in words:
    chars.update(list(w))

# Add start/end tokens
chars.add('<S>')
chars.add('<E>')

vocab = sorted(list(chars))
vocab_size = len(vocab)

# print("Vocab:", vocab)
# print("Vocab size:", vocab_size)

        ###

y= sorted(d.items(), key = lambda kv: -kv[1] )##sort by count.
#print(y)
N = torch.zeros((28, 28), dtype= torch.int32)
Chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(Chars)}##mapping from s to i               
stoi['<S>']= 26
stoi['<E>']= 27                                
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs , chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# print(N)
# Reverse index-to-char mapping
itos = {i: s for s, i in stoi.items()}

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

# Add character pairs and counts to the grid
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]  # Bigram string, e.g., 'th'
        plt.text(j, i, chstr, ha="center", va="bottom", color='grey', fontsize=8)
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='grey', fontsize=8)

plt.axis('off')
plt.title("Bigram Frequency Matrix", fontsize=20)
plt.tight_layout()

# Save to file instead of showing
plt.savefig("bigram_matrix1o.png", dpi=300)
print("Bigram matrix image saved as bigram_matrix1o0.png")


