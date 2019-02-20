from keras.preprocessing.text import Tokenizer


""""
Keras Tokenizer API
Keras provides a more sophisticated API for preparing text that can be fit and reused to prepare multiple text 
documents. This may be the preferred approach for large projects. Keras provides the Tokenizer class for preparing
text documents for deep learning. The Tokenizer must be constructed and then fit on either raw text documents 
or integer encoded text documents.
"""
# define 5 documents
docs = [
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!'
]

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)

""""
Once fit, the Tokenizer provides 4 attributes that you can use to query what has been learned about 
your documents:
    *word counts: A dictionary of words and their counts.
    *word docs: An integer count of the total number of documents that were used to fit the Tokenizer.
    *word index: A dictionary of words and their uniquely assigned integers.
    *document count: A dictionary of words and how many documents each appeared in.
"""
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

""""
Once the Tokenizer has been fit on training data, it can be used to encode documents in the train or test datasets.
The texts to matrix() function on the Tokenizer can be used to create one vector per document provided per input.
The length of the vectors is the total size of the vocabulary. This function provides a suite of standard 
bag-of-words model text encoding schemes that can be provided via a mode argument to the function. 
The modes available include:
    *binary: Whether or not each word is present in the document. This is the default.
    *count: The count of each word in the document.
    *tfidf: The Text Frequency-Inverse DocumentFrequency (TF-IDF) scoring for each word in the document.
    *freq: The frequency of each word as a ratio of words within each document.
"""
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)
