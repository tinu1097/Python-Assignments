import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ff_data = pd.read_csv("C:\\Users\\Hp\\Downloads\\pythone practice\\ANN\\forestfires.csv")
ff_data.drop(["month","day"],axis=1,inplace=True)
ff_data.dtypes
ff_data.shape
ff_data.columns
ff_data.isnull().sum()

#Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ff_data.iloc[:,0:28])
df_norm = pd.DataFrame(df_norm)
ff_norm = pd.concat([df_norm,ff_data.size_category],axis=1)
ff_norm = pd.DataFrame(ff_norm)

##convert string values into numerical ##
ff_norm.loc[ff_norm.size_category=="small","size_category"] = 0
ff_norm.loc[ff_norm.size_category=="large","size_category"] = 1
ff_norm.dtypes

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
ff_norm['size_category'] = LabelEncoder()
ff_norm.size_category = ff_norm.size_category.astype(object)
ff_norm.dtypes

# by value_counts we know how many number belong 0&1
ff_norm.size_category.value_counts()

x = ff_norm.drop(["size_category"],axis=1)
y = ff_norm["size_category"]
plt.hist(y)

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(20,20))
mlp.fit(x_train,y_train)
prediction_train=mlp.predict(x_train)
prediction_test = mlp.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,prediction_test))
y_test.value_counts()
pd.crosstab(y_test,prediction_test)
y_train.value_counts()
pd.crosstab(y_train,prediction_train)
np.mean(y_test==prediction_test)   ##0.73
np.mean(y_train==prediction_train) ##0.81































