import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
dibetes = pd.read_csv("C:\\Users\\Hp\\Downloads\\pythone practice\\naviye bayes\\Diabetes_RF.csv") 
dibetes.dtypes
dibetes.columns
col = list(dibetes.columns) 
inputs = col[:8]
output = col[8]
dx_train,dx_test,dy_train,dy_test = train_test_split(dibetes[inputs],dibetes[output],test_size = 0.2,random_state=0)

dg = GaussianNB()
mb = MultinomialNB()
dpred_gb = dg.fit(dx_train,dy_train).predict(dx_test)
dpred_mb = mb.fit(dx_train,dy_train).predict(dx_test)

# Confusion matrix 
pd.crosstab(dy_test,dpred_gb)
confusion_matrix(dy_test,dpred_gb) 
accuracy = sum(dpred_gb==dy_test)/dy_test.shape[0]
print ("Accuracy",(93+29)/(93+18+14+29)) # 76.19 

confusion_matrix(dy_test,dpred_mb)
print ("Accuracy",(76+22)/(76+31+25+22)) # 0.63



















