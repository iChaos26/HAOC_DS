#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import statsmodels.api as sm


#kaggle_separado = csv.reader(open('/Downloads/kaggle_separado'), delimiter=",")
pd_hertz= pd.read_excel("C:\Users\logonrmlocal\Downloads\Book1.xlsx")


# In[50]:


pd_hertz=pd_hertz.dropna()
pd_hertz.tail()


# In[29]:


X=pd_hertz.loc[:,['cigsPerDay']]
desvpad=11.920094 #amostral
ux=9 #hipotese nula
list_X=X.values.tolist()
n = len(list_X)
xb = np.mean(X)
desvpadb=desvpad/n**(0.5)
a = 0.05
X.describe()


# In[30]:



valor_p = (1-stats.norm.cdf(xb, loc=ux, scale=desvpadb))*2 # vezes 2 se for bicaudal, area de rejeição H0
print("valor-p = {0}\n".format(valor_p))

xcrit1 = stats.norm.ppf(a/2, loc=ux, scale=desvpadb)#esquerda
xcrit2 = stats.norm.ppf(1-a/2, loc=ux, scale=desvpadb)#direita

#pdf de Xb
x = np.arange(stats.norm.ppf(0.001, loc=ux, scale=desvpadb), stats.norm.ppf(0.999, loc=ux, scale=desvpadb), 0.01)
plt.plot(x, stats.norm.pdf(x, loc=ux, scale=desvpadb));

#RC1 - à esquerda
x = np.arange(stats.norm.ppf(0.001, loc=ux, scale=desvpadb), xcrit1, 0.01)
plt.fill_between(x, stats.norm.pdf(x, loc=ux, scale=desvpadb), color='red');
#RC2 - à direita
x = np.arange(xcrit2, stats.norm.ppf(0.999, loc=ux, scale=desvpadb), 0.01)
plt.fill_between(x, stats.norm.pdf(x, loc=ux, scale=desvpadb), color='red');

print("xcrit1 = {0}".format(xcrit1))
print("xcrit2 = {0}".format(xcrit2))


# In[33]:


X=pd_hertz.loc[:,['sysBP']]
desvpad=22.038097 #amostral
ux=132 #hipotese nula
list_X=X.values.tolist()
n = len(list_X)
xb = np.mean(X)
desvpadb=desvpad/n**(0.5)
a = 0.05
X.describe()


# In[41]:


xcrit1 = stats.norm.ppf(1-a, loc=ux, scale=desvpadb)# direita
xcrit2 = stats.norm.ppf(a, loc=ux, scale=desvpadb)#esquerda
print("xcrit = {0}".format(xcrit1))
valor_p = (1-stats.norm.cdf(xb, loc=ux, scale=desvpadb))
print("valor-p = {0}".format(valor_p))
print ("Nível de Significância em = {0}, acima dos 2%".format((valor_p)*100))


# In[51]:


import statsmodels.api as sm

Y = pd_hertz.loc[:,['sysBP']]
X = pd_hertz.loc[:,['cigsPerDay']]
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.summary()


# In[ ]:




