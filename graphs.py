# -*- coding: utf-8 -*-
"""Graphs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z46HhXsFKJZqQ3hKcULVXOp6LFpk_h3s
"""

import numpy as np
import matplotlib.pyplot as plt


# Data
x = np.arange(0, 10, 0.1)
print(x)
y = np.sin(x)
print(y)

# Graph creation
plt.plot(x,y)

# Determine chart properties
plt.title("Sine Graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

# show graph
plt.show()

import matplotlib.pyplot as plt
import numpy as np

#Data
x= np.random.rand(50)
y= np.random.rand(50)
colors= np.random.rand(50)
sizes= np.random.randint(50,150,50)

#Graph creation
plt.scatter(x,y, c=colors, s=sizes)

#Determine chart properties
plt.title("point chart")
plt.xlabel("x axis")
plt.ylabel("y axis31")
plt.show()

import matplotlib.pyplot as plt
import numpy as np

#Data
data = np.random.randn(50)

#Graph creation
plt.hist(data, bins=40)

#Determine chart properties
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency31")

plt.show()

import matplotlib.pyplot as plt
import numpy as np
#Data
x = ['A','B','C','D','E']
y= np.random.randint(1,15,5)

# Graph Creation
plt.bar(x,y)

# Determine chart properties
plt.title("Column Chart")
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.show()

import matplotlib.pyplot as plt

# Data
sizes = [31,25,15,10,59,5]

#Graph Creation
plt.pie(sizes)

#Determine chart prpoerties
plt.title("pie chart")

plt.show()