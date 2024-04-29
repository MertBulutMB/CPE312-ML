#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\advertising.csv")


# In[7]:


df


# In[ ]:


df=pd.read_csv("C:\\Users\\Mert\\Desktop\\ML\\data\\\linear_regression_dataset.csv")


# In[6]:


df


# In[9]:


# 'deneyim;maas' sütununu ';' karakterinden ayırarak yeni sütunlar oluşturma
df[['deneyim', 'maas']] = df['deneyim;maas'].str.split(';', expand=True)

# Yeni sütunların veri tiplerini uygun şekilde dönüştürme
df['deneyim'] = df['deneyim'].astype(float)
df['maas'] = df['maas'].astype(float)

# Scatter plot oluşturma
plt.scatter(df['deneyim'], df['maas'])
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()


# In[10]:


x=df.deneyim.values


# In[11]:


x.shape


# In[14]:


x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)


# In[15]:


from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(x,y)


# In[16]:


b0=linear_reg.predict([[0]])


# In[17]:


b0=linear_reg.intercept_
print(b0)


# In[18]:


b1=linear_reg.coef_
print(b1)


# In[19]:


new_salary=1663+1138*11
print(new_salary)


# In[21]:


b11=linear_reg.predict([[11]])
print(b11)


# In[22]:


y_head=linear_reg.predict(x)
plt.plot(x,y_head,color="red")
plt.scatter(x,y)
plt.show()


# In[23]:


from sklearn.metrics import r2_score
print("r square score",r2_score(y,y_head))


# In[24]:


from sklearn.metrics import mean_squared_error, r2_score

# Modelin tahminlerini alın
y_pred = linear_reg.predict(x)

# MSE hesaplayın
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

# RMSE hesaplayın
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared değerini hesaplayın
r_squared = r2_score(y, y_pred)
print("R-squared:", r_squared)

# Gerçek ve tahmin edilen değerlerin dağılımını görselleştirin
plt.scatter(y, y_pred)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek Değerler vs. Tahmin Edilen Değerler")
plt.show()


# In[25]:


# Hata dağılımını görselleştirin
plt.scatter(y_pred, y_pred - y, color='blue')
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linewidth=2)
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hata (Residuals)')
plt.title('Hata Dağılımı')
plt.show()


# In[26]:


from sklearn.model_selection import cross_val_score

# Çapraz doğrulama ile R-squared değerini hesaplayın
cv_scores = cross_val_score(linear_reg, x, y, cv=5)
print("Cross-Validation R-squared Scores:", cv_scores)
print("Ortalama R-squared:", np.mean(cv_scores))


# In[ ]:




