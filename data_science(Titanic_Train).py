#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 


# In[10]:


titanic_train = pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\train.csv")


# In[11]:


titanic_train.shape


# In[37]:


categorical = titanic_train.dtypes[titanic_train.dtypes == "object"].index
print(categorical)
titanic_train[categorical].describe()
titanic_train["Ticket"].describe()
del titanic_train["Ticket"]


# In[38]:


titanic_train.head()


# In[13]:


del titanic_train["PassengerId"]


# In[19]:


new_Pclass = pd.Categorical(titanic_train["Pclass"],ordered = True)

new_Pclass = new_Pclass.rename_categories(["class1","class2","class3"])

new_Pclass.describe()


# In[20]:


titanic_train["Age"].describe()


# In[23]:


missing = np.where(titanic_train["Age"].isnull() == True)
missing


# In[24]:


len(missing[0])


# In[25]:


titanic_train.hist(column="Age",figsize=(9,6),bins=20)


# In[27]:


new_age = np.where(titanic_train["Age"].isnull(), 28, titanic_train["Age"])
titanic_train["Age"]= new_age
titanic_train["Age"].describe()


# In[29]:


titanic_train.hist(column="Age",figsize=(9,6),bins=20)


# In[33]:


titanic_train["Pclass"] = new_Pclass
titanic_train["Cabin"].unique()
titanic_train.head()


# In[32]:


char_cabin = titanic_train["Cabin"].astype(str)
new_Cabin = np.array([cabin[0] for cabin in char_cabin])
new_Cabin = pd.Categorical(new_Cabin)
new_Cabin.describe()


# In[36]:


new_age_var = np.where(titanic_train["Age"].isnull(), 28, titanic_train["Age"])

titanic_train["Age"] = new_age_var
titanic_train["Age"].describe()


# In[ ]:




