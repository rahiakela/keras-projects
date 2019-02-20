from keras.preprocessing.text import text_to_word_sequence, one_hot


""""
Encoding with one hot
Keras provides the one hot() function that you can use to tokenize and integer encode a text document in one step.
We can use the text to word sequence() function from the previous section to split the document into words and
then use a set to represent only the unique words in the document.
The size of this set can be used to estimate the size of the vocabulary for one document.
"""
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

""""
We can put this together with the one hot() function and encode the words in the document.
The vocabulary size is increased by one-third to minimize collisions when hashing words.
"""
# integer encode the document
result = one_hot(text, round(vocab_size * 1.3))
print(result)
