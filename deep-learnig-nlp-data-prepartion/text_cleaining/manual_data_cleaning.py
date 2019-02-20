import re
import string


""""Manual Tokenization"""
""""
Text cleaning is hard, but the text we have chosen to work with is pretty clean already. We
could just write some Python code to clean it up manually, and this is a good exercise for those
simple problems that you encounter. Tools like regular expressions and splitting strings can get
you a long way.
"""

""""Load Data"""
filename = '../data/metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

""""Split by Whitespace"""
""""
Clean text often means a list of words or tokens that we can work with in our machine learning
models. This means converting the raw text into a list of words and saving it again. A very
simple way to do this would be to split the document by white space, including \ " (space), new
lines, tabs and more. We can do this in Python with the split() function on the loaded string.
"""
# split into words by white space
words = text.split()
print(words[:100])  # show first 100 word

""""Select Words"""
""""
Another approach might be to use the regex model (re) and split the document into words by
selecting for strings of alphanumeric characters (a-z, A-Z, 0-9 and ` ').
"""
# split based on words only
words = re.split(r'\W+', text)
print(words[:100])

""""Split by Whitespace and Remove Punctuation"""
""""
We may want the words, but without the punctuation like commas and quotes. We also want to
keep contractions together. One way would be to split the document into words by white space
(as in the section Split by Whitespace), then use string translation to replace all punctuation with
nothing (e.g. remove it). Python provides a constant called string.punctuation that provides a
great list of punctuation characters.
"""
print(string.punctuation)

""""
We can use regular expressions to select for the punctuation characters and use the sub()
function to replace them with nothing.
"""
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in words]
print(stripped[:100])

""""
Sometimes text data may contain non-printable characters. We can use a similar approach to
filter out all non-printable characters by selecting the inverse of the string.printable constant.
"""
re_print = re.compile('[^%s]' % re.escape(string.printable))
result = [re_print.sub('', w) for w in words]
print(result[:100])

""""Normalizing Case"""
""""
It is common to convert all words to one case. This means that the vocabulary will shrink in
size, but some distinctions are lost (e.g. Apple the company vs apple the fruit is a commonly
used example). We can convert all words to lowercase by calling the lower() function on each word.
"""
# convert to lower case
words = [word.lower() for word in words]
print(words[:100])

""""
Cleaning text is really hard, problem specific, and full of trade-offs. Remember, simple is better.
Simpler text data, simpler models, smaller vocabularies. You can always make things more
complex later to see if it results in better model skill.
"""
