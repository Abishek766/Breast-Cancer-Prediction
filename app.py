import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
import math
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def loadPage():
    return render_template("home.html", query="")

@app.route("/", methods=["POST"])
def cancerPrediction():
    url ="https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/refs/heads/master/breast-cancer-data.csv"
    df = pd.read_csv(url)
    inputQuery1 = request.form["query1"]
    inputQuery2 = request.form["query2"]
    inputQuery3 = request.form["query3"]
    inputQuery4 = request.form["query4"]
    inputQuery5 = request.form["query5"]
    inputQuery6 = request.form["query6"]
    inputQuery7 = request.form["query7"]
    inputQuery8 = request.form["query8"]
    inputQuery9 = request.form["query9"]
    inputQuery10 = request.form["query10"]

    label_encoding = LabelEncoder()
    df['diagnosis']=label_encoding.fit_transform(df.diagnosis.values)

    X = df[['radius_mean','perimeter_mean','area_mean','concavity_mean',
        'concave points_mean','radius_worst','perimeter_worst','area_worst','concavity_worst'
        ,'concave points_worst']]
    y = df['diagnosis']
    X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
    
    pca = PCA(n_components=5)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train_pca)
    X_test_sc = sc.transform(X_test_pca)

    lgm = LGBMClassifier()
    lgm.fit(X_train_sc,y_train)
    # y_pred_lgm=lgm.predict(X_test_sc)

    # new_data=[[17.99,122.8,1001,0.3001,0.1471,25.38,184.6,2019,0.7119,0.2654]]
    new_data =[[inputQuery1,inputQuery2,inputQuery3,inputQuery4,inputQuery5,
                inputQuery6,inputQuery7,inputQuery8,inputQuery9,inputQuery10]]
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

    return render_template("home.html", Output1= output1, Output2=output2, query1=request.form["query1"],
                           query2=request.form["query2"],query3=request.form["query3"],query4=request.form["query4"],
                           query5=request.form["query5"],query6=request.form["query6"],query7=request.form["query7"],
                           query8=request.form["query8"],query9=request.form["query9"],query10=request.form["query10"])

app.run()