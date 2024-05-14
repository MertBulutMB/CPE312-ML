#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn.metrics import confusion_matrix


# In[2]:


data = pd.read_csv(r"C:\Users\Mert\Desktop\ML\data\data (7).csv")


# In[3]:


data.info()
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


# In[4]:


data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]


# In[5]:


y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[6]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[7]:


x


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)


# In[9]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Decision Tree Score: " ,dt.score(x_test,y_test))


# In[10]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)


# In[11]:


print("Random Forest Algorithm result:",rf.score(x_test,y_test))


# In[13]:


# Decision Tree için confusion matrix
dt_cm = confusion_matrix(y_test, dt.predict(x_test))

# RandomForest için confusion matrix
rf_cm = confusion_matrix(y_test, y_pred)

# Confusion matrixleri görselleştirme
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(dt_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.subplot(1, 2, 2)
sns.heatmap(rf_cm, annot=True, cmap='Greens', fmt='d', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.show()

# RandomForest için özellik önem sıralaması
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]

# Özellik önem sıralamasını görselleştirme
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances[indices], y=x.columns[indices], palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.show()


# In[ ]:




