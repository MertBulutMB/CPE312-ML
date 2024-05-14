#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[22]:


data = pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data.csv")


# In[23]:


data.head()


# In[24]:


data.info()


# In[25]:


data.drop(["Unnamed: 32","id"],axis=1, inplace=True)


# In[26]:


M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]


# In[27]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant") 
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[28]:


data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis] 


# In[29]:


y = data.diagnosis.values 


# In[30]:


x_data= data.iloc[:,1:3].values 


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size=0.3,random_state=1)


# In[32]:


from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)


# In[33]:


from sklearn.tree import DecisionTreeClassifier
tree_classification=DecisionTreeClassifier(random_state=1,criterion='entropy')
tree_classification.fit(x_train,y_train)


# In[34]:


y_head=tree_classification.predict(x_test)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_head)
accuracy


# In[36]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_head)


# In[37]:


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,fmt='.0f',linewidths=0.5,linecolor="red",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_head")
plt.show()


# In[38]:


# Lojistik Regresyon modelini oluşturma
log_reg = LogisticRegression()

# Modeli eğitme
log_reg.fit(x_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred_log_reg = log_reg.predict(x_test)

# Modelin doğruluk skorunu hesaplama
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Lojistik Regresyon modelinin doğruluk skoru:", accuracy_log_reg)


# In[40]:


# Rastgele Ormanlar modelini oluşturma
rf_classifier = RandomForestClassifier(random_state=1)

# Modeli eğitme
rf_classifier.fit(x_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred_rf = rf_classifier.predict(x_test)

# Modelin doğruluk skorunu hesaplama
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Rastgele Ormanlar modelinin doğruluk skoru:", accuracy_rf)


# In[ ]:




