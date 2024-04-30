#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


data=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data(breastcancer).csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[6]:


M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]


# In[7]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.legend()
plt.xlabel("radius mean")
plt.ylabel("area mean")
plt.show()


# In[8]:


data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]


# In[9]:


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[10]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[11]:


x.head()


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[13]:


from sklearn.svm import SVC
svc=SVC(random_state=42)
svc.fit(x_train,y_train)


# In[14]:


svc.score(x_test,y_test)


# In[17]:


train_accuracy=[] # train_accuracy adında bir liste oluşturuluyor

test_accuracy=[]

for i in range(1,100):

    svm=SVC(C=i)

    svm.fit(x_train,y_train)

    train_accuracy.append(svm.score(x_train,y_train))

    test_accuracy.append(svm.score(x_test,y_test))

plt.plot(range(1,100),train_accuracy,label="training accuracy")

plt.plot(range(1,100),test_accuracy,label="testing accuracy")

plt.xlabel("c values")

plt.ylabel("accuracy")

plt.legend()

plt.show()


# In[18]:


from sklearn.model_selection import GridSearchCV

# GridSearchCV için parametre gridini tanımlayın
param_grid = {'C': [0.1, 1, 10, 100]}

# SVC modelini tanımlayın
svm = SVC(random_state=42)

# GridSearchCV nesnesini oluşturun
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)

# GridSearchCV'yi eğitin
grid_search.fit(x_train, y_train)

# En iyi parametreleri ve en iyi skoru gösterin
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)


# In[20]:


# En iyi C değerini kullanarak SVM modelini oluşturun
best_svm = SVC(C=grid_search.best_params_['C'], random_state=42)

# Modeli eğitin
best_svm.fit(x_train, y_train)

# Test verileri üzerinde modelin doğruluğunu değerlendirin
test_accuracy = best_svm.score(x_test, y_test)

print("Test doğruluğu:", test_accuracy)


# In[ ]:




