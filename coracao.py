#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 20:00:13 2019

@author: joao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
#O que quero prever? Idade x fatores. O ideal para ML é o contrário, tentar prever idade pelos indicadores 
#Data Preprocessing, não há dados categóricos 
dataset = pd.read_csv('heart.csv', sep=";")
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:13].values
#X = sm.add_constant(X)
#split into teste set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#fit do modelo. reshape de array para ndarray, pois X foi 
#importado como uma lista simples, e a regressão precisa de indices de valores [[X]]. 
#O retorno esperado é um indice, não o próprio valor
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#regressor.fit(np.array(X_train).reshape((X_train.shape[0], 1)), np.array(y_train))
#predição
y_pred = regressor.predict(X_test)
#y_pred = regressor.predict(np.array(X_train).reshape((X_train.shape[0], 1)))
#summary, NS = 0.1
import statsmodels.formula.api as sm
X= np.append(arr = np.ones((303, 1)).astype(int), values = X, axis = 1)
X_otimizada = X[:, [3,4,11,12]]
regressor_OLS = sm.OLS(endog = y, exog =X_otimizada).fit()
regressor_OLS.summary()
#feito a matriz otimizada, novo y_pred de idade
regressor.fit(X_otimizada, y_train)
y_pred_opt = regressor.predict(X_otimizada)