# -*- coding: utf-8 -*-
"""plotly_labdeneme.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17hbFv2EJtkbJpEFqf90BStbscX9QgJd7
"""

import plotly.express as px

fig = px.line(x=[1, 2, 3], y=[4, 5, 6])
fig.show()

df = px.data.iris()
fig = px.line(df, x="species", y="petal_length")
fig.show()

df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width")
fig.show()

df = px.data.iris()
fig = px.bar(df, x="species", y="petal_width")
fig.show()

df = px.data.tips()
fig = px.pie(df, values="total_bill", names="day")
fig.show()