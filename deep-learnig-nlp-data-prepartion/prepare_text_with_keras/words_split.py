from keras.preprocessing.text import text_to_word_sequence


""""
Split Words with text to word sequence
A good first step when working with text is to split it into words. Words are called to-
kens and the process of splitting text into tokens is called tokenization. Keras provides the
text to word sequence() function that you can use to split text into a list of words. By
default, this function automatically does 3 things:
    1-Splits words by space.
    2-Filters out punctuation.
    3-Converts text to lowercase (lower=True).
"""
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)
