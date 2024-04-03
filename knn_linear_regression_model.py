#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np



# In[3]:


iris = load_iris()


# In[4]:


x, y = iris.data, iris.target


# In[6]:


kf = KFold(n_splits=5, shuffle=True, random_state=64)  


# In[7]:


model = LinearRegression()


# In[9]:


scores = []
for train_index, test_index, in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = mean_squared_error(y_test, y_pred)
scores.append(score)


# In[10]:


mean_score = np.mean(scores)


# In[14]:


print("k fold cross validation result:", scores)

print("MSE:", mean_score)

