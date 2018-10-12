#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:05:47 2018

@author: yizhouwang
"""
#Muti Score


#Train -Test Split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
def DT(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train,y_train)
    param_range = {'max_depth':range(1,10,1)}
    gsearch = GridSearchCV(estimator = tree,
                            param_grid = param_range, 
                            scoring='accuracy',
                            cv=10)
    gsearch.fit(X_train,y_train)
    print(gsearch.best_params_) 
    print(gsearch.best_score_)
    y_train_pred = gsearch.predict(X_train)
    y_test_pred = gsearch.predict(X_test)
    
    print('Tree-train accuracy score: ',accuracy_score(y_train,y_train_pred))
    
    print('Tree-test accuracy score: ',accuracy_score(y_test,y_test_pred))
    

def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5, p=2,metric='minkowski')    
    knn.fit(X_train,y_train)
    param_range = {'n_neighbors':range(5,30,1)}
    gsearch = GridSearchCV(estimator = knn,
                            param_grid = param_range, 
                            scoring='accuracy',
                            cv=10)

    gsearch.fit(X_train, y_train)
    print(gsearch.best_params_) 
    print(gsearch.best_score_)
    gsearch.fit(X_train,y_train)
    y_train_pred = gsearch.predict(X_train)
    y_test_pred = gsearch.predict(X_test)
   
    print('Knn-train accuracy score: ',accuracy_score(y_train,y_train_pred))
    
    print('Knn-test accuracy score: ',accuracy_score(y_test,y_test_pred))
    
def SVM(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear', random_state=1, C=1.0)  
    svm.fit(X_train,y_train)
    param_range = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    gsearch = GridSearchCV(estimator = svm,
                            param_grid = param_range, 
                            scoring='accuracy',
                            cv=10)
    gsearch.fit(X_train,y_train)
    print(gsearch.best_params_) 
    print(gsearch.best_score_)
    y_train_pred = gsearch.predict(X_train)
    y_test_pred = gsearch.predict(X_test)
   
    print('svm-train accuracy score: ',accuracy_score(y_train,y_train_pred))
    
    print('svm-test accuracy score: ',accuracy_score(y_test,y_test_pred))
    
DT(X_train, X_test, y_train, y_test)
KNN(X_train, X_test, y_train, y_test)
SVM(X_train, X_test, y_train, y_test)