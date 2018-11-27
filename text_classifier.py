#importing library
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#Method 1
#importing dataset
reviews = load_files('txt_sentoken/');
X,y = reviews.data, reviews.target

#Method 2
#Storing as pickle Files
with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)

with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)

# Unpickle Files
with open('X.pickle', 'rb') as f:
    X=pickle.load(f)

with open('y.pickle', 'rb') as f:
    y=pickle.load(f)

#Preprocessing
#Creating the corpus(list of doc)
corpus = []
for i range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)


#Bag of Words
#vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words = stopwords.words('english'))
#X = vectorizer.fit_transform(corpus).toarray()

#TF-IDF Model
#transformer = TfidfTransformer()
#X = transformer.fit_transform(X).toarray()

#Direct TF_IDF Model
vectorizer = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


#Divide train-test data
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size=0.2, random_state = 0)


#ML Logistic Regression
"""
The sentiment analysis task is mainly a binary classification
problem to perfect whether a given sentence is positive of negative.
In our demonstrations we denote '0' as negative and '1' as positive.
"""

#Point Concepts

#1. Each sentence is mapped to a point.
#2. If the point is greater than 0.5 then positive else negative.

classifier = LogisticRegression()
classifier.fit(text_train, sent_train)

sent_pred = classifier.predict(text_test)

cm = confusion_matrix(sent_test, sent_pred)

#cm[0][0] + cm[1][1])/4 (339 out of 400)

#Saving Classifier Model
with open('classifier.pickle', 'wb') as f:
    pickle.dumb(classifier,f)

#Saving the vectorizer
with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dumb(vectorizer,f)
