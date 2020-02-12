import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
company = pd.read_csv("C:\\Users\\Hp\\Downloads\\pythone practice\\Company_Data.csv")
company.isnull().sum()
company.dtypes
#company.Education = company.Education.astype(object)
company['Sales'].dtype
###convert into categorical values###
company['Sales'][company['Sales']>=9.5]= 10
company['Sales'][company['Sales']<9.5]= 0
company['Sales'][company['Sales']==10]= 'Yes'
company['Sales'][company['Sales']==0]= 'No'
company['Sales'] = company['Sales'].astype('object')
#company['CompPrice'] = company['CompPrice'].astype('float64')
#company['Income'] = company['Income'].astype('float64')
#######################################
company.dtypes
company['Sales'].unique()
company.Sales.value_counts()
##central of tradency$###
company.Income.mean()
company.Income.median()
#company.Income.mode()
company.Price.mean()
company.Price.median()
#company.Price.mode()
company.Population.mean()
company.Population.median()
#company.Population.mode()
company.Sales.mode()
###spliting data traina nd test##
from sklearn.model_selection import train_test_split
com_train,com_test = train_test_split(company,test_size = 0.3)
company_input = com_train.iloc[:,1:]
company_output = com_train.iloc[:,0]
#company_output = pd.DataFrame(company_output)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
company_input['ShelveLoc'] = label_encoder.fit_transform(company_input['ShelveLoc'])
company_input['Urban'] = label_encoder.fit_transform(company_input['Urban'])
company_input['US'] = label_encoder.fit_transform(company_input['US'])
####################################
company_in = com_test.iloc[:,1:]
company_out = com_test.iloc[:,0]
#company_out = pd.DataFrame(company_out)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
company_in['ShelveLoc'] = label_encoder.fit_transform(company_in['ShelveLoc'])
company_in['Urban'] = label_encoder.fit_transform(company_in['Urban'])
company_in['US'] = label_encoder.fit_transform(company_in['US'])

from sklearn.tree import DecisionTreeClassifier
com_model = DecisionTreeClassifier(criterion = 'entropy')
com_model.fit(company_input,company_output)

predict_test = com_model.predict(company_in)
pd.Series(predict_test).value_counts()
pd.crosstab(company_out,predict_test)
temp = pd.Series(com_model.predict(company_in)).reset_index(drop=True)
# Accuracy = train
np.mean(com_train.Sales == com_model.predict(company_input)) ##100%
# Accuracy = Test
np.mean(predict_test==com_test.Sales) # 75%%











































