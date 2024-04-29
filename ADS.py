#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\advertising.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x = df[['TV','Radio','Newspaper']]
y=df['Sales']


# In[9]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=100)


# In[10]:


from sklearn.linear_model import LinearRegression
mlr=LinearRegression()
mlr.fit(x_train,y_train)


# In[11]:


print("Intercept:", mlr.intercept_)


# In[12]:


print("Coefficents:", list(zip(x, mlr.coef_)))


# In[13]:


y_pred = mlr.predict(x_test)
print("prediction for test set: {}".format(y_pred))


# In[14]:


mlr_dff = pd.DataFrame({"Actual values": y_test, "Prediciton values":y_pred})
mlr_dff.head()


# In[15]:


from sklearn import metrics 
meansqerr = metrics.mean_squared_error(y_test,y_pred)
print("R squared: {:.2f}".format(mlr.score(x,y)*100))
print("Mean Squared error:", meansqerr)


# In[16]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# In[17]:


plt.hist(y_test - y_pred, bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.show()


# In[19]:


coefficients = pd.DataFrame({"Feature": x.columns, "Coefficient": mlr.coef_})
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.show()


# In[ ]:




