"""" Hashing with HashingVectorizer """
""""
Counts and frequencies can be very useful, but one limitation of these methods is that the
vocabulary can become very large. This, in turn, will require large vectors for encoding
documents and impose large requirements on memory and slow down algorithms. A clever work
around is to use a one way hash of words to convert them to integers. The clever part is that
no vocabulary is required and you can choose an arbitrary-long fixed length vector. A downside
is that the hash is a one-way function so there is no way to convert the encoding back to a word.
The HashingVectorizer class implements this approach that can be used to consistently
hash words, then tokenize and encode documents as needed.
An arbitrary fixed-length vector sizeof 20 was chosen. This corresponds to the range of the hash function, 
where small values (like 20) may result in hash collisions.
"""
from sklearn.feature_extraction.text import HashingVectorizer


# list of text documents
text = ['The quick brown fox jumped over the lazy dog.']

# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

