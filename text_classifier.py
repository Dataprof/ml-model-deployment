# Databricks notebook source
!pip install nltk

# COMMAND ----------

import nltk
import numpy as np
import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/Restaurant_Reviews.tsv.txt',delimiter='\t',quoting=3)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
nltk.download('stopwords')
import re
for i in range(0,len(dataset)):
  customer_review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  customer_review = customer_review.lower()
  customer_review = customer_review.split()
  clean_reviw = [ps.stem(word) for word in customer_review if not word in set(stopwords.words('english'))]
  clean_reviw = ' '.join(clean_reviw)
  corpus.append(clean_reviw)
corpus[6]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1500,min_df=3,max_df=0.6)
X= vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)  
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifierKNN.fit(X_train,y_train)
y_pred = classifierKNN.predict(X_test)
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(y_test,y_pred)
sample = ['Good  batting by england']
sample = vectorizer.transform(sample).toarray()
sentiment = classifierKNN.predict(sample)
sentiment
import pickle
with open('textclassifier.pkl','wb') as f:
  pickle.dump(classifierKNN,f)
with open('tfidfmodel.pkl','wb') as f:
  pickle.dump(vectorizer,f)

# COMMAND ----------

