#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data.csv")


# In[5]:


data.head()


# In[6]:


data.info()


# In[9]:


data.drop(["Unnamed: 32","id"], axis=1, inplace=True)


# In[10]:


M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]


# In[11]:


M.info()


# In[24]:


plt.scatter(M.radius_mean,M.area_mean,color="red",label="malignant")
plt.scatter(M.radius_mean,M.area_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.show()


# In[17]:


plt.scatter(M.radius_mean,M.texture_mean,color="green",label="malignant")
plt.scatter(M.radius_mean,M.texture_mean,color="blue",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.show()


# In[25]:


data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]


# In[26]:


data.diagnosis


# In[27]:


y= data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# In[28]:


x = (x_data - np.min(x_data))/ (np.max(x_data))- (np.min(x_data))


# In[29]:


from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.3, random_state=1)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)


# In[ ]:




