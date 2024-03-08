#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np


# In[2]:


array = np.array([1,2,3])
print(array)


# In[8]:


array2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(array2.shape)


# In[9]:


a = array2.reshape(3,5)
print(a)


# In[10]:


print("Shape:",a.shape)


# In[11]:


print("Dimentsion:",a.ndim)


# In[12]:


print("Data Type:",a.dtype.name)


# In[14]:


array2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print("Vector:",array2)
a=array2.reshape(3,5)
print("Two dimensional array:",a)


# In[15]:


print("shape:",a.shape)
print("dimension:,a.ndim")
print("data type:",a.dtype.name)
print("size:,a.size")
print("type:",type(a))


# In[16]:


x=np.array([1,2,3])
y=np.array([4,5,6])
print(x+y)
print(x-y)


# In[17]:


a=np.array([1,2,3,4,5,6,7])
print(a[0])
print(a[0:4])


# In[18]:


reverse_array=a[::-1]
print(reverse_array)


# In[19]:


a=np.array([1,2,3])
d=a.copy()
print(d)
b=a
c=a
b[0]=5
print(a,b,c)


# In[20]:


a=np.array([1,2,3,4,5,6,7])
print(a[0])
print(a[0:4])


# In[21]:


reverse_array=a[::-1]
print(reverse_array)


# In[26]:


b=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(b)
print(b[1,1])
print(b[:,1])
print(b[1,:])
print(b[1,1:4])
print(b[-1,:])
print(b[:,-1])


# In[8]:


array = np.array([1,2,3])
print(array)


# In[10]:


array2 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(array2.shape)


# In[11]:


a = array2.reshape(3,5)
print(a)


# In[12]:


print("Shape:",a.shape)


# In[13]:


print("Dimension:",a.ndim)


# In[14]:


print("Data Type:",a.dtype.name)


# In[15]:


print("size",a.size)


# In[16]:


print("type:",type(a))


# In[21]:


array3 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(array3)


# In[22]:


array3.shape


# In[25]:


zeros = np.zeros((3,4))
print(zeros)


# In[26]:


zeros[0,0]=5
print(zeros)


# In[27]:


ones = np.ones((3,4))
print(ones)


# In[28]:


np.empty((4,5))


# In[29]:


np.arange(10,50,5)


# In[30]:


a = np.linspace(0,10,20)
print(a)


# In[33]:


a = np.array([75,13,31])
b = np.array([41,39,26])
print(a+b)
print(a-b)


# In[34]:


liste = [23,31,45,29]
array = np.array([35,71,56,19])
print(liste)
print(array)


# In[35]:


a = np.array([1,2,3])


# In[36]:


b=a 
c=a


# In[37]:


b[0]=5
print(a,b,c)


# In[38]:


a[0]=9
d = a.copy()
print(d)


# In[39]:


a = np.array([1,2,3,4,5,6,7,8,9,10])
print(a[5])


# In[41]:


a[::-1]


# ##### a[0:4]

# In[ ]:


b = np.array([1,2,3,4])

