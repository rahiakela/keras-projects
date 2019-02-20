from keras.preprocessing.text import text_to_word_sequence, hashing_trick


""""
Hash Encoding with hashing trick
A limitation of integer and count base encodings is that they must maintain a vocabulary of words and their mapping
to integers. An alternative to this approach is to use a one-way hash function to convert words to integers.
This avoids the need to keep track of a vocabulary, which is faster and requires less memory.
Keras provides the hashing trick() function that tokenizes and then integer encodes the document, just like
the one hot() function. It provides more exibility, allowing you to specify the hash function as either
hash (the default) or other hash functions such as the built in md5 function or your own function.
"""
# define the document
text = 'The quick brown fox jumped over the lazy dog.'

# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

# integer encode the document
result = hashing_trick(text, round(vocab_size * 1.3), hash_function='md5')
print(result)
