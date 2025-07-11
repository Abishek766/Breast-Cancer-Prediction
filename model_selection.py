import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

url ="https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/refs/heads/master/breast-cancer-data.csv"
df = pd.read_csv(url)

# print(df.head(5)) 

#Checking the distribution of Y Variable
print(df.diagnosis.value_counts())

# B    357
# M    212

print(df.diagnosis.value_counts()/len(df)*100)

# B    62.741652  - 62%
# M    37.258348  - 37%

# print(df.info())

#Unnamed values are present in dataset
#check the null values

# print(df['Unnamed: 32'].isnull().sum()) #569
#Check the non-null values

# print(df['Unnamed: 32'].notnull().sum()) #0

# This column is not useful for prediction, so we neglected it
df = df.drop('Unnamed: 32', axis=1)
# df.info()

#Feature encoding for labeled variables
label_encoding = LabelEncoder()
df['diagnosis']=label_encoding.fit_transform(df.diagnosis.values)


#Feature Selection
X = df[['radius_mean','perimeter_mean','area_mean','concavity_mean',
        'concave points_mean','radius_worst','perimeter_worst','area_worst','concavity_worst'
        ,'concave points_worst']]
# print(X.columns)

y = df['diagnosis']

#Spliting the data into training and testing
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

#Bulid the Base Model RandomForest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Accuracy Score Random Forest: ", accuracy_score(y_test,y_pred_rf)) # 0.95

# XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)

print("Accuracy Score XGBoost: ",accuracy_score(y_test,y_pred_xgb)) # 0.93

# LightGBM
lgm = LGBMClassifier()
lgm.fit(X_train,y_train)
y_pred_lgm=lgm.predict(X_test)

print("Accuracy score LightGBM: ", accuracy_score(y_test,y_pred_lgm)) # 0.96

#Comparing this three models with base parameters we find LightGBM is a best fit for this dataset
# with 0.96 accuracy


