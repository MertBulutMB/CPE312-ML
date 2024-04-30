#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


data=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data.csv")


# In[5]:


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[6]:


data.tail()


# In[7]:


M= data[data.diagnosis =="M"]
B= data[data.diagnosis =="B"]


# In[8]:


plt.scatter(M.radius_mean,M.texture_mean,color="blue",label="malign",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="purple",label="benign",alpha=0.3)
plt.xlabel("radius of the tumor")
plt.ylabel("texture of the tumor")
plt.legend()
plt.show()


# In[9]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y= data.diagnosis.values
x_data= data.drop(["diagnosis"],axis=1)


# In[10]:


x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)


# In[12]:


from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
nb.fit(x_train,y_train)


# In[13]:


GaussianNB()


# In[14]:


print("accuracy of svm algorithm: ",nb.score(x_test,y_test))


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix

# Test seti üzerinde tahmin yapma
y_pred = nb.predict(x_test)

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Sınıflandırma raporu oluşturma
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)


# In[16]:


import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[17]:


from sklearn.metrics import roc_curve, roc_auc_score

# Sınıf olasılıklarını al
y_pred_prob = nb.predict_proba(x_test)[:, 1]

# ROC eğrisini oluştur
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# ROC AUC skorunu hesapla
roc_auc = roc_auc_score(y_test, y_pred_prob)
print('ROC AUC Score:', roc_auc)


# In[ ]:




