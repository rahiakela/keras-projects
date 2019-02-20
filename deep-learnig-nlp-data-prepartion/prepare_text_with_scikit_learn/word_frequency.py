"""" Word Frequencies with TfidfVectorizer """
""""
Word counts are a good starting point, but are very basic. One issue with simple counts is that
some words like the will appear many times and their large counts will not be very meaningful
in the encoded vectors. An alternative is to calculate word frequencies, and by far the most
popular method is called TF-IDF. This is an acronym that stands for Term Frequency - Inverse
Document Frequency which are the components of the resulting scores assigned to each word.
    -Term Frequency: This summarizes how often a given word appears within a document.
    -Inverse Document Frequency: This downscales words that appear a lot across documents.
Without going into the math, TF-IDF are word frequency scores that try to highlight
words that are more interesting, e.g. frequent in a document but not across documents.
The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document
frequency weightings, and allow you to encode new documents.    
"""
from sklearn.feature_extraction.text import TfidfVectorizer


# list of text documents
text = [
    'The quick brown fox jumped over the lazy dog.',
    'The dog.',
    'The fox'
]

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
