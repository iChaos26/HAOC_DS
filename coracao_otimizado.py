#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:40:20 2019

@author: joao
"""

#feito a otimização da matriz X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
dataset = pd.read_csv('heart.csv', sep=";")
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, [3,4,11,12]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#fit do modelo. reshape de array para ndarray, pois X foi 
#importado como uma lista simples, e a regressão precisa de indices de valores [[X]]. 
#O retorno esperado é um indice, não o próprio valor
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
regressor_OLS = sm.OLS(endog = y, exog =X).fit()
regressor_OLS.summary()