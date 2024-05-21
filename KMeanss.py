#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
os.environ["OMP_NUM_THREADS"] = "1"


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[13]:


dataset = pd.read_csv("C:/Users/Mert/Desktop/ML/data/Mall_Customers.csv")


# In[4]:


dataset.head(10)


# In[5]:


dataset.shape


# In[6]:


dataset.info()


# In[7]:


dataset.isnull().sum()


# In[8]:


x=dataset.iloc[:,[3,4]].values


# In[23]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[15]:


plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no of cluster")
plt.ylabel("wcss")
plt.show()


# In[20]:


kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)


# In[17]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c="red",label="cluster 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c="blue",label="cluster 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c="green",label="cluster 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c="cyan",label="cluster 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c="magenta",label="cluster 5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="centroids")
plt.title("müşteri classları")
plt.xlabel("yıllık gelir")
plt.ylabel("harcama skoru")
plt.legend()
plt.show()


# In[ ]:




