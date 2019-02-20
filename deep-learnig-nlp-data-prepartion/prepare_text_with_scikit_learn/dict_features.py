"""" Loading features from dicts """
""""
The class DictVectorizer can be used to convert feature arrays represented as lists of standard Python 
dict objects to the NumPy/SciPy representation used by scikit-learn estimators.
While not particularly fast to process, Python’s dict has the advantages of being convenient to use, 
being sparse (absent features need not be stored) and storing feature names in addition to values.
DictVectorizer implements what is called one-of-K or “one-hot” coding for categorical (aka nominal, discrete) features.
Categorical features are “attribute-value” pairs where the value is restricted to a list of discrete of 
possibilities without ordering (e.g. topic identifiers, types of objects, tags, names…).
"""
from sklearn.feature_extraction import DictVectorizer


# In the following, “city” is a categorical attribute while “temperature” is a traditional numerical feature
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]

# create transform
vectorizer = DictVectorizer()
vectorizer.fit_transform(measurements).toarray()
print(vectorizer.get_feature_names())
