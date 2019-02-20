""""
A pipeline of text preparation including:
1- Load the raw text.
2- Split into tokens.
3- Convert to lowercase.
4- Remove punctuation from each token.
5- Filter out remaining tokens that are not alphabetic.
6- Filter out tokens that are stop words.
7- Optional: reducing each word to its root or base using Stemming.
"""
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

# 1- Load the raw text.
filename = '../data/metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

# 2- Split into tokens.
tokens = word_tokenize(text)

# 3- Convert to lowercase.
tokens = [w.lower() for w in tokens]

# 4- Remove punctuation from each token.
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
stripped = [re_punc.sub('', w) for w in tokens]

# 5- Filter out remaining tokens that are not alphabetic.
words = [word for word in stripped if word.isalpha()]

# 6- Filter out tokens that are stop words.
stop_words = set(stopwords.words('english'))
words = [w for w in words if w not in stop_words]
print(words[:100])

# 7- Optional: reducing each word to its root or base using Stemming.
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in words]
print(stemmed[:100])
