import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

url ="https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/refs/heads/master/breast-cancer-data.csv"
df = pd.read_csv(url)

# print(df.head(5)) 

#Checking the distribution of Y Variable
print(df.diagnosis.value_counts())

# B    357
# M    212

print(df.diagnosis.value_counts()/len(df)*100)

# B    62.741652  - 62% - Bengin - Non-cancerous
# M    37.258348  - 37% - Malignant - Cancerous

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

#Bulid the Base Model LightGBM
# lgm = LGBMClassifier()
# lgm.fit(X_train,y_train)
# y_pred_lgm=lgm.predict(X_test)

# print("Accuracy score LightGBM: ", accuracy_score(y_test,y_pred_lgm)) # 0.96

#Feature Engineering
#Feature Extraction Using PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#Feature Extraction using LDA
# lda = LinearDiscriminantAnalysis()
# X_train_lda = lda.fit_transform(X_train, y_train)
# X_test_lda = lda.transform(X_test)

# def best_model(model,X_train,X_test,y_train,y_test):
#     base_model=model
#     base_model.fit(X_train,y_train)
#     y_predict=base_model.predict(X_test)
#     print("Accuracy score of the model: ", accuracy_score(y_test,y_predict))

# best_model(LGBMClassifier(verbose=0), X_train_pca, X_test_pca, y_train, y_test) 
# # IN PCA with 5 features we have 0.96 accuracy
# # with 3 features we have 0.95 accuracy

# best_model(LGBMClassifier(verbose=0), X_train_lda,X_test_lda,y_train, y_test)
# IN LDA with 5 features we have 0.94 accuracy

#So we go with pca with 5 features for feature Extractions

#Feature Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_pca)
X_test_sc = sc.transform(X_test_pca)

lgm = LGBMClassifier()
lgm.fit(X_train_sc,y_train)
y_pred_lgm=lgm.predict(X_test_sc)

print("Accuracy score LightGBM: ", accuracy_score(y_test,y_pred_lgm)) #0.96

#Testing with Random Data points
new_data=[[17.99,122.8,1001,0.3001,0.1471,25.38,184.6,2019,0.7119,0.2654]]
new_df=pd.DataFrame(new_data, columns=['radius_mean','perimeter_mean','area_mean','concavity_mean',
        'concave points_mean','radius_worst','perimeter_worst','area_worst','concavity_worst'
        ,'concave points_worst'])

new_pca = pca.transform(new_df)
new_sc = sc.transform(new_pca)
single=lgm.predict(new_sc)
proba = lgm.predict_proba(new_sc)[:,1]

if single == 1:
    output1="The patient is affected with Breast Cancer with confidence"
    output2="Confidence: {}",format(proba*100)

else:
    output1="The patient is not affected with Breast Cancer with confidence"
    output2=""