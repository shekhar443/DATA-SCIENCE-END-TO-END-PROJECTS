#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_excel("C:/Users/USER/Desktop/heroku/firstcsv.xlsx")


# In[4]:


df.head()


# In[5]:


df.isnull().any()


# In[6]:


gender_df=pd.get_dummies(df['Gender'],drop_first=True)
gender_df


# In[7]:


df.drop('User ID',axis=1,inplace=True)


# In[8]:


df


# In[9]:


df.drop('Gender',axis=1,inplace=True)


# In[10]:


df


# In[11]:


df=pd.concat([df,gender_df],axis=1)


# In[12]:


df.head()


# In[13]:


X = df.iloc[:, [0, 1,3]].values
X


# In[14]:


y = df.iloc[:, -2].values
y


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[16]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:





# In[17]:


# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[18]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[19]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
ac


# In[20]:


import pickle
pickle.dump(sc, open("scaler.pickle", "wb"))
ssc = pickle.load(open("scaler.pickle", 'rb')) 

pickle.dump(classifier, open('nbclassifier.pkl','wb'))

model = pickle.load(open('nbclassifier.pkl','rb'))


# In[21]:


sample=ssc.transform([X_test[0]])
sample


# In[22]:


model.predict(sample)


# In[23]:


y_test[0]


# In[24]:


cm = confusion_matrix(y_test, y_pred)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




