""" Word Counts with CountVectorizer """
"""
The CountVectorizer provides a simple way to both tokenize a collection of text documents
and build a vocabulary of known words, but also to encode new documents using that vocabulary.
You can use it as follows:
    1- Create an instance of the CountVectorizer class.
    2- Call the fit() function in order to learn a vocabulary from one or more documents.
    3- Call the transform() function on one or more documents as needed to encode each as a vector.
An encoded vector is returned with a length of the entire vocabulary and an integer count
for the number of times each word appeared in the document. Because these vectors will
contain a lot of zeros, we call them sparse. Python provides an efficient way of handling sparse
vectors in the scipy.sparse package. The vectors returned from a call to transform() will
be sparse vectors, and you can transform them back to NumPy arrays to look and better
understand what is going on by calling the toarray() function.    
"""
from sklearn.feature_extraction.text import CountVectorizer


# list of text documents
text = ['The quick brown fox jumped over the lazy dog.']

# create the transform
vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform(text)

# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

# encode another document
text2 = ['the puppy']
vector = vectorizer.transform(text2)
print(vector.toarray())
