#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

labels=["Setosa","Versicolor","Virginica"]
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
plt.figure(figsize=(8,6))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()


# In[19]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

x, y = make_classification(n_samples=2000, n_classes=2, random_state=55)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=150)
clf = LogisticRegression()
clf.fit(x_train, y_train)

y_pred= clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)
recall= tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision + recall)
accuracy = (tp+tn) / (tp+tn+fp+fn)

plt.figure(figsize=(8,6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Negative", "Positive"])
plt.yticks(tick_marks, ["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.text(0,0, f"True Negative: {tn}", ha="center", color="yellow", fontsize=14)
plt.text(0,1, f"False Negative: {fn}", ha="center", color="yellow", fontsize=14)
plt.text(1,0, f"False Positive: {fp}", ha="center", color="yellow", fontsize=14)
plt.text(1,1, f"True Positive: {tp}", ha="center", color="yellow", fontsize=14)

plt.text(2.5,0, f"Precision: {precision:.2f}", ha="center", va="center" ,color="black", fontsize=14)
plt.text(2.5,-0.2, f"Recall: {recall:.2f}", ha="center", va="center" ,color="black", fontsize=14)
plt.text(2.5,-0.4, f"F1 Score: {f1_score:.2f}", ha="center", va="center" ,color="black", fontsize=14)
plt.text(2.5,-0.6, f"Accuracy: {accuracy:.2f}", ha="center", va="center" ,color="black", fontsize=14)
plt.show()


# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Iris dataseti üzerinde bir model oluşturma
iris = load_iris()
x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
clf_iris = DecisionTreeClassifier()
clf_iris.fit(x_train_iris, y_train_iris)
y_pred_iris = clf_iris.predict(x_test_iris)
cm_iris = confusion_matrix(y_test_iris, y_pred_iris)

plt.figure(figsize=(8, 6))
labels = ["Setosa", "Versicolor", "Virginica"]
df_cm_iris = pd.DataFrame(cm_iris, index=labels, columns=labels)
sns.heatmap(df_cm_iris, annot=True, cmap=plt.cm.Blues)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Iris Dataseti Üzerindeki Decision Tree Modelinin Confusion Matrixi")
plt.show()

# Yapay veri üzerinde bir model oluşturma
x, y = make_classification(n_samples=2000, n_classes=2, random_state=55)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=150)
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Yapay Veri Üzerindeki Logistic Regression Modelinin Confusion Matrixi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Negative", "Positive"])
plt.yticks(tick_marks, ["Negative", "Positive"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (tp + tn) / (tp + tn + fp + fn)

plt.text(0, 0, f"True Negative: {tn}", ha="center", color="yellow", fontsize=14)
plt.text(0, 1, f"False Negative: {fn}", ha="center", color="yellow", fontsize=14)
plt.text(1, 0, f"False Positive: {fp}", ha="center", color="yellow", fontsize=14)
plt.text(1, 1, f"True Positive: {tp}", ha="center", color="yellow", fontsize=14)

plt.text(2.5, 0, f"Precision: {precision:.2f}", ha="center", va="center", color="black", fontsize=14)
plt.text(2.5, -0.2, f"Recall: {recall:.2f}", ha="center", va="center", color="black", fontsize=14)
plt.text(2.5, -0.4, f"F1 Score: {f1_score:.2f}", ha="center", va="center", color="black", fontsize=14)
plt.text(2.5, -0.6, f"Accuracy: {accuracy:.2f}", ha="center", va="center", color="black", fontsize=14)

plt.show()


# In[3]:






# In[ ]:




