"""
Name : Ankan Dash
CS 675 Introduction to Machine Learning, NJIT
Project : Spam vs Ham prediction
"""

import os
import sys
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# training and testing data
train = sys.argv[1]
test = sys.argv[2]

# reading and processing the training data
train_data = pd.read_csv(train, encoding="ISO-8859-1")
train_data = train_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
train_data.columns = ['label', 'message']
train_data['length'] = train_data['message'].apply(len)

# importing some NLP libraries
import string
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# function to clean the data 
def text_processing(text):
  '''
  1. remove punctuations
  2. remove stopwords
  3. return the processed text
  '''
  no_punc = [char for char in text if char not in string.punctuation]

  no_punc = ''.join(no_punc)

  return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

# importing some sklearn text processing libraries and classification models
from sklearn.feature_extraction.text import CountVectorizer

bag_of_word = CountVectorizer(analyzer=text_processing).fit(train_data['message'])

df_bag_of_word = bag_of_word.transform(train_data['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer().fit(df_bag_of_word)

df_tfidf = tfidf.transform(df_bag_of_word)

from sklearn.svm import LinearSVC

svc_model = LinearSVC().fit(df_tfidf,train_data['label'])

# reading and processing the testing data and predicting the labels for the test data
test_data = pd.read_csv(test, encoding="ISO-8859-1")
test_data = test_data.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis=1)
test_data.columns = ['label', 'message']
test_data['length'] = test_data['message'].apply(len)

df_test_bag_of_word = bag_of_word.transform(test_data['message'])
df_test_tfidf = tfidf.transform(df_test_bag_of_word)
predictions = svc_model.predict(df_test_tfidf)
true_label = test_data['label']

# printing the model predictions
for pred in predictions:
	print(pred)
print('\n')

# model evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(true_label, predictions))
print('\n')
print(classification_report(true_label, predictions))
print('\n')
print('Accuracy: ',round(accuracy_score(true_label, predictions)*100,2), " %")