# BigramNames
uses simple bigram model to make the matrix
Here is a **guided README.md** tailored for learners, explaining both the **core bigram concept** and the **structure of your Python file** step-by-step with educational comments.

---

# 📘 Bigram Name Model – Guided Learning Project

This is a beginner-friendly project that teaches you how to build and visualize a **bigram frequency model** from a list of names using **Python** and **PyTorch**.

---

## 💡 What is a Bigram?

A **bigram** is a pair of consecutive elements from a sequence. In natural language processing (NLP), it typically refers to two consecutive characters or words.

For example:

* The word `chat` contains the bigrams:
  `(<S>, c), (c, h), (h, a), (a, t), (t, <E>)`
  (`<S>` and `<E>` are special tokens added to mark the start and end of a word.)

We use bigrams to analyze which character tends to follow another. This is a fundamental idea behind language models and text generation.

---

## 🛠️ Requirements

* Python 3.x
* PyTorch
* Matplotlib

Install with:

```bash
pip install torch matplotlib
```

---

## File Structure

* `names.txt`: A plain text file with one name per line.
* `bigram_model.py`: The main script. Builds and visualizes the bigram matrix.


---

## Key Concepts Covered

### Reading and Tokenizing Words

We read all names from `names.txt` and convert each word into a sequence of characters, adding `<S>` (start) and `<E>` (end) tokens to each:

```python
chs = ['<S>'] + list(w) + ['<E>']
```

---

### Building a Dictionary of Bigram Counts

We use a Python dictionary to count how often each character pair appears:

```python
d = {}
for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    d[bigram] = d.get(bigram, 0) + 1
```

---

### Character Vocabulary Creation

We build a vocabulary of all unique characters in the dataset:

```python
chars = set()
for w in words:
    chars.update(list(w))
chars.add('<S>')
chars.add('<E>')
```

This allows us to convert characters to indexes for matrix operations.

---

### Bigram Count Matrix with PyTorch

We build a 2D matrix `N` of shape `[vocab_size x vocab_size]` where `N[i][j]` stores the number of times character `i` is followed by character `j`.

```python
N = torch.zeros((28, 28), dtype=torch.int32)
```

Characters are mapped to indices using:

```python
stoi = {s: i for i, s in enumerate(Chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
```

Each bigram updates the count in the matrix:

```python
N[ix1, ix2] += 1
```

---

### Visualization

We use `matplotlib` to visualize the matrix:

```python
plt.imshow(N, cmap='Blues')
```

Each cell displays:

* The bigram (e.g., "th")
* The count (e.g., 12)

```python
plt.text(j, i, chstr, ha="center", va="bottom", color='grey', fontsize=8)
plt.text(j, i, N[i, j].item(), ha="center", va="top", color='grey', fontsize=8)
```

Finally, the plot is saved to a file:

```python
plt.savefig("bigram_matrix1o.png", dpi=300)
```

---

## Output

The output is a heatmap showing how frequently each character pair appears in the names dataset. Darker cells represent higher frequency.

Example:

|     | a  | b  | c | ... | <E> |
| --- | -- | -- | - | --- | --- |
| <S> | 30 | 5  | 0 | ... | 0   |
| a   | 0  | 12 | 3 | ... | 10  |
| ... |    |    |   |     |     |

---

## Educational Tips

* Try printing `d` or `y` to see which bigrams are most common.
* Modify `names.txt` to test how the model changes.
* Normalize `N` to convert it into a probability matrix.
* Use the bigram matrix to generate new words.

---

## Next Steps

* Convert counts to probabilities with Laplace smoothing.
* Extend this to **trigrams** or even a **neural network language model**.
* Try generating new names based on the bigram probabilities.

---

## Author

Created as part of an educational series for learning NLP fundamentals and PyTorch. Inspired by Andrej Karpathy’s *makemore* series.

---

decoded by aditya panigrahi
