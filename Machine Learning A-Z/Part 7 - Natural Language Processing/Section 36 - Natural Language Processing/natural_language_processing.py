''' Natural Language Processing
We use tsv files (tab-separated values) instead of csv files (comma separated values).
This avoids ambiguity errors with the delimiter since reviews can contain commas while
its rare for them to have tabs.

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)#quoting =3, ignores double quotes in reviews

# Cleaning the texts
''' This removes words that do not help the algoithm to cassify the review e.g articles,
conjunctions.
    It also removes punctuation marks.
    Stemming; this obtains the root tense of a word e.g loved to love so to reduce number 
of words.
    Get rid of uppercase letters
    
'''
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][0])  #remove character that are not letters
    review = review.lower()                                 #lower case
    review = review.split()                                 #split words
    ps = PorterStemmer()
    review = [word for word in review if not word in set(stopwords.words('english'))]  #remove stopwords & stem
    corpus.append(review)
    
    
# Creating the Bag of Words model
'''create a sparse matrix (matrix with many zeros,) each word appears as a column, the rows are the 
reviews and each cell is the number of times a word appears in a review'''
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)        #max features removes less commom words, this kepps 1500 most frequent words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

'''At this point we have a classification problem which we can solve by any of the classification,
algorithms we looked st earlier, for NLP, the most most commonly used are Naive bayes, decision tree 
and random forest.'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)