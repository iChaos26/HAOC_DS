#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:01:53 2019

@author: joao
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
dataset = pd.read_csv('heart.csv', sep=";")
y = dataset.iloc[:, 0].values
#plot de 3 regressões
X_trestb = dataset.iloc[:, [4]].values
X_train_trestb, X_test_trestb, y_train, y_test = train_test_split(X_trestb, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train_trestb, y_train)
y_pred = regressor.predict(X_test_trestb)
#Plot do trestb por colesterol
#plot training set
plt.scatter(X_train_trestb, y_train, c='green')
plt.plot(X_train_trestb, regressor.predict(X_train_trestb), c='black')
plt.title('Idade(Training Set)')
plt.xlabel('Blood Pressure')
plt.show()
#plot test set
plt.scatter(X_test_trestb, y_test, c='orange')
plt.plot(X_train_trestb, regressor.predict(X_train_trestb), c='black')
plt.title('Colesterol (Test Set)')
plt.xlabel('Blood Pressure')
plt.show()