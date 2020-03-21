#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


data = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Real Estate Price Prediction\\Real estate.csv')
data.head()


# In[7]:


data.describe()


# In[19]:


x = data.iloc[:,1:-1]
y = data.iloc[:,7]
x.isnull().sum()


# In[ ]:





# In[17]:


plt.scatter(x['X3 distance to the nearest MRT station'],y)
plt.show()


# In[16]:


plt.scatter(x['X2 house age'],y)
plt.xlabel('House Age')
plt.ylabel('House Price Unit Area')
plt.show()


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[16]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)


# In[20]:


y_predict = regression.predict(x_test)
y_predict
regression.score(x_test,y_test)


# In[18]:


from sklearn.metrics import r2_score
score = r2_score(y_test,y_predict)
score


# In[25]:


from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor()
rfreg.fit(x_train,y_train)
rfreg.score(x_test,y_test)

