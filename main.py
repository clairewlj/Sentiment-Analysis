# coding: utf-8

# In[3]:

import re
# import csv
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.classify.scikitlearn import SklearnClassifier

# In[90]:

# pandas read csv
# read data
train_df = pd.read_csv("/Users/ClaireWang/Documents/study/previous/STOR 767/train.csv", header=0, encoding='ISO-8859-1')
test_df = pd.read_csv("/Users/ClaireWang/Documents/study/previous/STOR 767/test.csv", header=0, encoding='ISO-8859-1')
# train_df.head(5)
# test_df.head(5)

train_df.columns = ['text', 'sentiment']
test_df.columns = ['text', 'sentiment']

# In[77]:

# data processing - tokenization, stop words removal, non-letters removal, stemming

# Pandas DataFrame columns are Pandas Series when you pull them out,
# which you can then call .tolist() on to turn them into a Python list
train_text = train_df.text.tolist()
test_text = test_df.text.tolist()

train_sentiment = train_df.sentiment
test_sentiment = test_df.sentiment

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # remove non letters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems


for i in range(len(train_text)):
    train_text[i] = train_text[i].lstrip('"').rstrip().lower()
    train_text[i] = ' '.join(tokenize(train_text[i]))

# In[76]:

# #f read
# #read data
# with open("/Users/ClaireWang/Documents/study/previous/STOR 767/train.csv",encoding="ISO-8859-1") as f:
#     f_train = f.readlines()[1:]
# with open("/Users/ClaireWang/Documents/study/previous/STOR 767/test.csv",encoding="ISO-8859-1") as f:
#     f_test = f.readlines()[1:]


# In[78]:

# bag-of-words
# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=tokenize,
    lowercase=True,
    stop_words='english',
    min_df=5
    # max_df=0.8
)

# In[107]:

# tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_t = TfidfVectorizer(
    min_df=5,
    stop_words='english',
    lowercase=True,
    tokenizer=tokenize,
    analyzer='word',
    sublinear_tf=True
    # max_df = 0.8
)

# In[98]:

# transform data into feature vectors & fit the model
corpus = train_text
corpus2 = test_text

# The method fit_transform does two functions: First, it fits the model and learns the vocabulary;
# second, it transforms our corpus data into feature vectors.
# The input to fit_transform should be a list of strings, so we concatenate train and test data as follows.

# CountVectorizer
corpus_data_features = vectorizer.fit_transform(corpus)
corpus_data_features2 = vectorizer.transform(corpus2)
# Numpy arrays are easy to work with, so convert the result to an array.
corpus_data_features_nd = corpus_data_features.toarray()
corpus_data_features_nd2 = corpus_data_features2.toarray()

# In[108]:

# TfIDF
corpus_data_features_t = vectorizer_t.fit_transform(corpus)
corpus_data_features2_t = vectorizer_t.transform(corpus2)
# Numpy arrays are easy to work with, so convert the result to an array.
corpus_data_features_nd_t = corpus_data_features_t.toarray()
corpus_data_features_nd2_t = corpus_data_features2_t.toarray()

# In[85]:

# check features and the counts
# vectorizer._validate_vocabulary()
vocab = vectorizer.get_feature_names()
# Sum up the counts of each vocabulary word
dist = np.sum(corpus_data_features_nd, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the data set
for tag, count in zip(vocab, dist):
    print(count, tag)

# In[102]:

# Logistic Regression - CountVectorizer
# train classifer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import time

log_model = LogisticRegression()
t0 = time.time()
log_model = log_model.fit(X=corpus_data_features_nd, y=train_sentiment)
t1 = time.time()
# label evaluation set.We can use either predict for classes or predict_proba for probabilities.
y_pred = log_model.predict(corpus_data_features_nd2)
t2 = time.time()
time_lr_train = t1 - t0
time_lr_predict = t2 - t1

# There is a function for classification called sklearn.metrics.classification_report
# which calculates several types of (predictive) scores on a classification model.

print("Results for Logistic Regression - CountVectorizer")
print("Training time: %fs; Prediction time: %fs" % (time_lr_train, time_lr_predict))
print(classification_report(test_sentiment, y_pred))

# In[109]:

# Logistic Regression - TFIDF Vectorizer
# train classifer
log_model = LogisticRegression()
t0 = time.time()
log_model = log_model.fit(X=corpus_data_features_nd_t, y=train_sentiment)
t1 = time.time()
# label evaluation set.We can use either predict for classes or predict_proba for probabilities.
y_pred = log_model.predict(corpus_data_features_nd2_t)
t2 = time.time()
time_lr_train = t1 - t0
time_lr_predict = t2 - t1

# There is a function for classification called sklearn.metrics.classification_report
# which calculates several types of (predictive) scores on a classification model.

print("Results for Logistic Regression - TFIDF")
print("Training time: %fs; Prediction time: %fs" % (time_lr_train, time_lr_predict))
print(classification_report(test_sentiment, y_pred))

# In[110]:

# SVM, kernel=RBF - CountVectorizer
from sklearn import svm

CountV_train = corpus_data_features_nd
CountV_test = corpus_data_features_nd2

# SVM with kernel = RBF
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(CountV_train, train_df.sentiment)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(CountV_test)
t2 = time.time()
time_rbf_train = t1 - t0
time_rbf_predict = t2 - t1

# Print results
print("Results for SVC(kernel=rbf) - CountVectorizer")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_sentiment, prediction_rbf))

# In[111]:

# SVM, kernel=RBF - TFIDF Vectorizer
CountV_train_t = corpus_data_features_nd_t
CountV_test_t = corpus_data_features_nd2_t

# SVM with kernel = RBF
classifier_rbf = svm.SVC()
t0 = time.time()
classifier_rbf.fit(CountV_train_t, train_df.sentiment)
t1 = time.time()
prediction_rbf = classifier_rbf.predict(CountV_test_t)
t2 = time.time()
time_rbf_train = t1 - t0
time_rbf_predict = t2 - t1

# Print results
print("Results for SVC(kernel=rbf) - TFIDF")
print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
print(classification_report(test_sentiment, prediction_rbf))

# In[103]:

# SVM with kernel = linear 1
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(CountV_train, train_df.sentiment)
t1 = time.time()
prediction_linear = classifier_linear.predict(CountV_test)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_sentiment, prediction_linear))

# In[112]:

# SVM with kernel = linear 1 - TFIDF
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(CountV_train_t, train_df.sentiment)
t1 = time.time()
prediction_linear = classifier_linear.predict(CountV_test_t)
t2 = time.time()
time_linear_train = t1 - t0
time_linear_predict = t2 - t1

print("Results for SVC(kernel=linear) - TFIDF")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(test_sentiment, prediction_linear))

# In[104]:

# SVM with kernel = linear 2
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(CountV_train, train_df.sentiment)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(CountV_test)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1

print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_sentiment, prediction_liblinear))

# In[113]:

# SVM with kernel = linear 2 - TFIDF
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(CountV_train_t, train_df.sentiment)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(CountV_test_t)
t2 = time.time()
time_liblinear_train = t1 - t0
time_liblinear_predict = t2 - t1

print("Results for LinearSVC() - TFIDF")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_sentiment, prediction_liblinear))

# In[105]:

# Gaussian Naive Bayes - CountVectorizer
from sklearn.naive_bayes import GaussianNB

classifier_gnb = GaussianNB()
t0 = time.time()
classifier_gnb.fit(CountV_train, train_sentiment)
t1 = time.time()
prediction_gnb = classifier_gnb.predict(CountV_test)
t2 = time.time()
time_gnb_train = t1 - t0
time_gnb_predict = t2 - t1

print("Results for Gaussian Naive Bayes")
print("Training time: %fs; Prediction time: %fs" % (time_gnb_train, time_gnb_predict))
print(classification_report(test_sentiment, prediction_gnb))

# In[114]:

# Gaussian Naive Bayes - TFIDF
from sklearn.naive_bayes import GaussianNB

classifier_gnb = GaussianNB()
t0 = time.time()
classifier_gnb.fit(CountV_train_t, train_sentiment)
t1 = time.time()
prediction_gnb = classifier_gnb.predict(CountV_test_t)
t2 = time.time()
time_gnb_train = t1 - t0
time_gnb_predict = t2 - t1

print("Results for Gaussian Naive Bayes - TFIDF")
print("Training time: %fs; Prediction time: %fs" % (time_gnb_train, time_gnb_predict))
print(classification_report(test_sentiment, prediction_gnb))

