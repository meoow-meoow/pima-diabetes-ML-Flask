import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as s
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def data_cleaning():
    #checking for null
    data.isnull().sum()

def train_test_data(data):
    #train_test_splitting of the dataset
    x = data.drop(columns = 'Outcome')

    # Getting target value
    y = data['Outcome']    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test

def LogisticRegressionModel(x_train,x_test,y_train,y_test,features):
    reg = LogisticRegression()
    reg.fit(x_train,y_train)                         
    y_pred=reg.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",reg.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    # return accuracy_score(y_test,y_pred)*100
    final_features = [np.array(features)]
    prediction = reg.predict(final_features)
    print(prediction)
    if prediction==0:
        result="You Are Non-Diabetic"
    else:
        result="You Are Diabetic"
    return result

def KNeighboursClassifierModel(x_train,x_test,y_train,y_test):
    knn=KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",knn.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def SVCModel(x_train,x_test,y_train,y_test):
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred=svc.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",svc.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def NaiveBayesModel(x_train,x_test,y_train,y_test):
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",gnb.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def DecisionTreeClassifierModel(x_train,x_test,y_train,y_test):
    dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')
    dtree.fit(x_train,y_train)
    y_pred=dtree.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",dtree.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def RandomForestClassifierModel(x_train,x_test,y_train,y_test):
    rfc=RandomForestClassifier()
    rfc.fit(x_train,y_train)
    y_pred=rfc.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",rfc.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def AdaBoostClassifierModel(x_train,x_test,y_train,y_test):
    adb = AdaBoostClassifier()
    adb.fit(x_train,y_train)
    y_pred=adb.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",adb.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def GradientBoostingClassifierModel(x_train,x_test,y_train,y_test):
    gbc=GradientBoostingClassifier()
    gbc.fit(x_train,y_train)
    y_pred=gbc.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",gbc.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def XGBClassifierModel(x_train,x_test,y_train,y_test):
    xgb =XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
    xgb.fit(x_train, y_train)
    y_pred=xgb.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",xgb.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

def ExtraTreeClassifierModel(x_train,x_test,y_train,y_test):
    etc= ExtraTreesClassifier(n_estimators=100, random_state=0)
    etc.fit(x_train,y_train)
    y_pred=etc.predict(x_test)
    # print("Classification Report is:\n",classification_report(y_test,y_pred))
    # print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    # print("Training Score:\n",etc.score(x_train,y_train)*100)
    # print("Mean Squared Error:\n",mean_squared_error(y_test,y_pred))
    # print("R2 score is:\n",r2_score(y_test,y_pred))
    return accuracy_score(y_test,y_pred)*100

if __name__ == "__main__":
    data= pd.read_csv("diabetes.csv")
    x_train,x_test,y_train,y_test = train_test_data(data)
    d = {}
    features = [1,	85,	66,	29,	0, 26.6, 0.351,	31]
    d['LogisticRegressionModel']= LogisticRegressionModel(x_train,x_test,y_train,y_test,features)
    # d['KNeighboursClassifierModel']= KNeighboursClassifierModel(x_train,x_test,y_train,y_test)
    # d['SVCModel']= SVCModel(x_train,x_test,y_train,y_test)
    # d['NaiveBayesModel']= NaiveBayesModel(x_train,x_test,y_train,y_test)
    # d['DecisionTreeClassifierModel']= DecisionTreeClassifierModel(x_train,x_test,y_train,y_test)
    # d['RandomForestClassifierModel']= RandomForestClassifierModel(x_train,x_test,y_train,y_test)
    # d['AdaBoostClassifierModel']= AdaBoostClassifierModel(x_train,x_test,y_train,y_test)
    # d['GradientBoostingClassifierModel']= GradientBoostingClassifierModel(x_train,x_test,y_train,y_test)
    # d['XGBClassifierModel']= XGBClassifierModel(x_train,x_test,y_train,y_test)
    # d['ExtraTreeClassifierModel']= ExtraTreeClassifierModel(x_train,x_test,y_train,y_test)
    print(d)

    


