#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.drop(["Unnamed: 32","id"],axis=1,inplace=True)


# In[6]:


M =data[data.diagnosis=="M"]
B =data[data.diagnosis=="B"]


# In[7]:


M.info()


# In[8]:


B.info()


# In[9]:


plt.scatter(M.radius_mean,M.area_mean,color="yellow",label="malignant")
plt.scatter(B.radius_mean,B.area_mean,color="blue",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("area_mean")
plt.show()


# In[10]:


plt.scatter(M.radius_mean,M.texture_mean,color="blue",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="yellow",label="benign")
plt.legend()
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()


# In[11]:


y=data.diagnosis.values


# In[12]:


x_data=data.iloc[:,1:3].values


# In[13]:


x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[14]:


x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[17]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[16]:


KNeighborsClassifier(n_neighbors=3)


# In[19]:


knn.fit(x_train, y_train)


# In[20]:


y = data["diagnosis"].values
x = data.iloc[:, 2:32].values


# In[21]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Veri setini yükle
data = pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\data.csv")

# Unnamed ve id sütunlarını düşür
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# M ve B tanımla
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# Eğitim ve test setlerini oluştur
x_data = data.iloc[:, 1:3].values
y = data["diagnosis"].values
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# KNN modelini oluştur ve eğit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)


# In[23]:


from sklearn.metrics import accuracy_score

# Test verileri üzerinde tahmin yap
y_pred = knn.predict(x_test)

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Test verileri üzerinde doğruluk:", accuracy)


# In[24]:


for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("k =", k, "için doğruluk:", accuracy)


# In[25]:


# Veri setindeki özellikler arasındaki ilişkileri görselleştirme
sns.pairplot(data, hue="diagnosis", vars=["radius_mean", "texture_mean", "area_mean", "smoothness_mean"])
plt.show()


# In[26]:


# Korelasyon matrisini oluşturma
correlation_matrix = data.corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()


# In[ ]:




