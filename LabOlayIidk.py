#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import warnings
import category_encoders as ce
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")


# In[13]:


data = r"C:\Users\Mert\Desktop\ML\data\car_evaluation.csv"

df = pd.read_csv(data, header=None)

# İlk 5 satırı göster
print(df.head())


# In[14]:


df.shape


# In[15]:


df.head()


# In[16]:


col_names = ["buying","maint","doors","person","lug_boot", "safety", "class"]
df.columns = col_names


# In[17]:


df.head()


# In[18]:


df.info()


# In[19]:


for col in col_names:
  print(df[col].value_counts())


# In[20]:


df["class"].value_counts()


# In[21]:


df.isnull().sum()


# In[22]:


x = df.drop(["class"],axis=1)


# In[23]:


y = df["class"]


# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)


# In[25]:


x_train.shape


# In[26]:


x_test.shape


# In[27]:


x_train.dtypes


# In[28]:


x_train.head()


# In[34]:


import category_encoders as ce


# In[35]:


encoder = ce.OrdinalEncoder(cols=["buying","maint","doors","person","lug_boot", "safety"])
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)


# In[36]:


x_train.head()


# In[37]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,random_state=0)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)


# In[38]:


from sklearn.metrics import accuracy_score
print("Model Accuracy: {0:0.4f}".format(accuracy_score(y_test,y_pred)))


# In[39]:


rfc_100 = RandomForestClassifier(n_estimators=100,random_state=0)
rfc_100.fit(x_train,y_train)
y_pred_100 = rfc_100.predict(x_test)
print("Model Accuracy with  100 decision tree: {0:0.4f}".format(accuracy_score(y_test,y_pred_100)))


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
rfc_100.fit(x_train, y_train)
y_pred_100 = rfc_100.predict(x_test)
print("Model Accuracy with 100 decision trees: {0:0.4f}".format(accuracy_score(y_test, y_pred_100)))


# In[ ]:




