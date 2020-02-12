

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

sal_train = pd.read_csv("C:\\Users\\Hp\\Downloads\\pythone practice\\SVM PY\\SalaryData_Train(1).csv")
sal_test = pd.read_csv("C:\\Users\\Hp\\Downloads\\pythone practice\\SVM PY\\SalaryData_Test(1).csv")

####train data set EDA ###
sal_train.dtypes
sal_train.columns
sal_train.isnull().sum()
sal_train.describe
sal_train.mean()
sal_train.median()
sal_train.shape
sal_train.Salary.value_counts()

###
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
sal_train['Salary'] = label_encoder.fit_transform(sal_train['Salary'])
sal_train.Salary = sal_train.Salary.astype('int64')
sal_train.Salary.value_counts()
#sal_train.Salary.astype('int')

sal_train['native'] = label_encoder.fit_transform(sal_train['native'])
sal_train.native= sal_train.native.astype(object)

sal_train['educationno'] = label_encoder.fit_transform(sal_train['educationno'])
sal_train.educationno= sal_train.educationno.astype(object)

sal_train['sex'] = label_encoder.fit_transform(sal_train['sex'])
sal_train.sex = sal_train.sex.astype(object)

sal_train['maritalstatus'] = label_encoder.fit_transform(sal_train['maritalstatus'])
sal_train.maritalstatus= sal_train.maritalstatus.astype(object)

sal_train['race'] = label_encoder.fit_transform(sal_train['race'])
sal_train.race= sal_train.race.astype(object)

sal_train['occupation'] = label_encoder.fit_transform(sal_train['occupation'])
sal_train.occupation= sal_train.occupation.astype(object)

sal_train['relationship'] = label_encoder.fit_transform(sal_train['relationship'])
sal_train.relationship= sal_train.relationship.astype(object)

sal_train['education'] = label_encoder.fit_transform(sal_train['education'])
sal_train.education= sal_train.education.astype(object)

sal_train['workclass'] = label_encoder.fit_transform(sal_train['workclass'])
sal_train.workclass= sal_train.workclass.astype(object)



sal_train.dtypes
plt.hist(sal_train.Salary)
sns.boxplot(x="Salary",y="age",data=sal_train,palette = "hls")

##test data EDA #####
sal_test.dtypes
sal_test.columns
sal_test.isnull().sum()
sal_test.describe
sal_test.mean()
sal_test.median()
sal_test.shape
sal_test.Salary.value_counts()
####################################
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
sal_test['Salary'] = label_encoder.fit_transform(sal_test['Salary'])
sal_test.Salary = sal_test.Salary.astype('int64')
sal_test.Salary.value_counts()
#sal_test.Salary.astype('int')

sal_test['native'] = label_encoder.fit_transform(sal_test['native'])
sal_test.native= sal_test.native.astype("object")

sal_test['educationno'] = label_encoder.fit_transform(sal_test['educationno'])
sal_test.educationno= sal_test.educationno.astype(object)

sal_test['sex'] = label_encoder.fit_transform(sal_test['sex'])
sal_test.sex = sal_test.sex.astype(object)

sal_test['maritalstatus'] = label_encoder.fit_transform(sal_test['maritalstatus'])
sal_test.maritalstatus= sal_test.maritalstatus.astype(object)

sal_test['race'] = label_encoder.fit_transform(sal_test['race'])
sal_test.race= sal_test.race.astype(object)

sal_test['occupation'] = label_encoder.fit_transform(sal_test['occupation'])
sal_test.occupation= sal_test.occupation.astype(object)

sal_test['relationship'] = label_encoder.fit_transform(sal_test['relationship'])
sal_test.relationship= sal_test.relationship.astype(object)

sal_test['education'] = label_encoder.fit_transform(sal_test['education'])
sal_test.education= sal_test.education.astype(object)

sal_test['workclass'] = label_encoder.fit_transform(sal_test['workclass'])
sal_test.workclass= sal_test.workclass.astype(object)

sal_test.dtypes
plt.hist(sal_test.Salary)
sns.boxplot(x="Salary",y="age",data=sal_test,palette = "hls")



###### for SVM ###
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train = sal_train.drop(["Salary"],axis=1)
y_train = sal_train["Salary"]
x_test = sal_test.drop(["Salary"],axis=1)
y_test = sal_test["Salary"]
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict()
###Accuracy##
np.mean(pred_test_linear==y_test)

#company_out = pd.DataFrame(company_out)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
company_in['ShelveLoc'] = label_encoder.fit_transform(['ShelveLoc'])
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)























