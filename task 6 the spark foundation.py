#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics
# TASK 6 - Prediction using Decision Tree Algorithm
# 
# Author : G Madhu chandana
# 
# ● Create the Decision Tree classifier and visualize it graphically.
# 
# ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.
# 
# Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


# data = pd.read_csv( r"C:\Users\USER\Downloads\Iris (1).csv")
data.head()


# In[4]:


data.describe()


# # Handling missing data

# In[5]:


data.isnull().sum()


# In[6]:


SepalLengthCm = data['SepalLengthCm']
SepalWidthCm = data['SepalWidthCm']
PetalLengthCm = data['PetalLengthCm']
PetalWidthCm = data['PetalWidthCm']

columns = [SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]

fig,ax = plt.subplots()
ax.boxplot(columns,notch=True,patch_artist=True)
plt.xticks([1,2,3,4],['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
plt.show()


# In[7]:


Q1 = data['SepalWidthCm'].quantile(0.25)
Q3 = data['SepalWidthCm'].quantile(0.75)
IQR = Q3 - Q1

ur = Q3+1.5*IQR
lr = Q1-1.5*IQR

samp = data.index[data['SepalWidthCm'] > ur]
samp.append(data.index[data['SepalWidthCm'] < lr])
data = data.drop(samp)
data.reset_index(drop=True)


# # Split the dataset for training and testing

# In[8]:


df = data.copy()
x = df.iloc[:,1:5]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[10]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[11]:


accuracy = accuracy_score(y_test,y_pred)
accuracy*100


# In[12]:


data = {'y_Actual': y_test,
        'y_Predicted': y_pred
        }

df = pd.DataFrame(data)
df.reset_index(inplace = True, drop = True)
df.head()


# In[13]:


confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)


# In[14]:


sns.heatmap(confusion_matrix, annot=True)
plt.show()


# In[15]:


from sklearn import tree
feature_names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300,facecolor='k')
tree.plot_tree(clf,feature_names=feature_names,class_names=class_names,filled=True);
fig.savefig('IrisTree.png')


# In[16]:


clf.predict_proba([[4.7,3.2,1.3,0.2]])


# In[17]:


clf.predict([[4.7,3.2,1.3,0.2]])

Hence, the Decision Tree Model has been created and visualized with the accuracy of 97.29% in the Test dataset. It also predict the class of the new data successfully.