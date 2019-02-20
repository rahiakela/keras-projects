import nltk
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

""""Tokenization and Cleaning with NLTK"""
""""
The Natural Language Toolkit, or NLTK for short, is a Python library written for working and
modeling text. It provides good tools for loading and cleaning text that we can use to get our
data ready for working with machine learning and deep learning algorithms.
"""

"""Split into Sentences"""
""""
A good useful first step is to split the text into sentences. Some modeling tasks prefer input
to be in the form of paragraphs or sentences, such as Word2Vec. You could first split your
text into sentences, split each sentence into words, then save each sentence to file, one per line.
NLTK provides the sent tokenize() function to split text into sentences.
"""
# load data
filename = '../data/metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# split into sentences
sentences = sent_tokenize(text)
print(sentences[0]) # show first sentence.

""""Split into Words"""
""""
NLTK provides a function called word tokenize() for splitting strings into tokens (nominally
words). It splits tokens based on white space and punctuation.
"""
# split into words
tokens = word_tokenize(text)
print(tokens[:100])  # show first 100 words

""""Filter Out Punctuation"""
""""
We can filter out all tokens that we are not interested in, such as all standalone punctuation. This
can be done by iterating over all tokens and only keeping those tokens that are all alphabetic.
Python has the function isalpha() that can be used.
"""
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])  # show first 100 words

""""Filter out Stop Words (and Pipeline)"""
""""
Stop words are those words that do not contribute to the deeper meaning of the phrase. They
are the most common words such as: the, a, and is. For some applications like documentation
classification, it may make sense to remove stop words. NLTK provides a list of commonly
agreed upon stop words for a variety of languages, such as English.
"""
# load the stop words of english
stop_words = stopwords.words('english')
print(stop_words)

""""Stem Words"""
"""
Stemming refers to the process of reducing each word to its root or base. For example fishing,
fished, fisher all reduce to the stem fish. Some applications, like document classification, may
benefit from stemming in order to both reduce the vocabulary and to focus on the sense or
sentiment of a document rather than deeper meaning. There are many stemming algorithms,
although a popular and long-standing method is the Porter Stemming algorithm. This method
is available in NLTK via the PorterStemmer class.
"""
# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])
