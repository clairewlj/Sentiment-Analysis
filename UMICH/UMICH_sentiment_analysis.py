
# coding: utf-8

# In[2]:

#This project is based on movie reviews on IMDb.
import urllib.request
import pandas as pd

#define URLs
test_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

#define local file names
test_data_file_name = 'test_data.csv'
train_data_file_name = 'train_data.csv'

#download files using urlib
test_data_f = urllib.request.urlretrieve(test_data_url,test_data_file_name)
train_data_f = urllib.request.urlretrieve(train_data_url,train_data_file_name)

#read and load files into data frames for processing
test_data_df = pd.read_csv(test_data_file_name,header=None,delimiter="\t",quoting=3)
test_data_df.columns = ['Text']
train_data_df = pd.read_csv(train_data_file_name,header=None,delimiter='\t',quoting=3)
train_data_df.columns = ['Sentiment','Text']


# In[3]:

#calculate average length of text
import numpy as np
np.mean([len(s.split(" ")) for s in train_data_df.Text])


# In[24]:

import re,nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

#######
#based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

#download for stopwords list
#nltk.download()

stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    #remove non letters
    text = re.sub('[^a-zA-Z]',' ',text)
    #tokenize
    tokens = nltk.word_tokenize(text)
    #stem
    stems = stem_tokens(tokens,stemmer)
    return stems

#######

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    max_features = 85
)


# In[25]:

#Pandas DataFrame columns are Pandas Series when you pull them out, which you can then call .tolist() on to turn them into a Python list
#The method fit_transform does two functions: First, it fits the model and learns the vocabulary; 
#second, it transforms our corpus data into feature vectors. 
#The input to fit_transform should be a list of strings, so we concatenate train and test data as follows.
corpus = train_data_df.Text.tolist()+test_data_df.Text.tolist()
corpus_data_features = vectorizer.fit_transform(corpus)


# In[26]:

#Numpy arrays are easy to work with, so convert the result to an array.
corpus_data_features_nd = corpus_data_features.toarray()


# In[27]:

vocab = vectorizer.get_feature_names()
# Sum up the counts of each vocabulary word
dist = np.sum(corpus_data_features_nd, axis=0)
    
# For each, print the vocabulary word and the number of times it 
# appears in the data set
#for tag, count in zip(vocab, dist):
#    print(count, tag)


# In[28]:

#create a separate evaluation test set from training data
from sklearn.model_selection import train_test_split
# remember that corpus_data_features_nd contains all of our original train and test data, so we need to exclude
# the unlabeled test entries

X_train, X_test, y_train,y_test = train_test_split(
    corpus_data_features_nd[0:len(train_data_df)],
    train_data_df.Sentiment,
    train_size = 0.85,
    random_state = 1234)


# In[29]:

#train classifer
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train,y=y_train)

#label evaluation set.We can use either predict for classes or predict_proba for probabilities.
y_pred = log_model.predict(X_test)

#There is a function for classification called sklearn.metrics.classification_report
#which calculates several types of (predictive) scores on a classification model.
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:



