#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[8]:


df = pd.read_csv('spam.csv', encoding='latin-1')
df.head()


# In[9]:


print(df)


# In[10]:


data = df.where((pd.notnull(df)), '')


# In[12]:


data.head(10)


# In[13]:


data.info()


# In[14]:


data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2" : "Message", "v1":"Category"})


# In[15]:


data.shape


# In[20]:


data.loc[data['Category']=='spam', 'Category',]= 0
data.loc[data['Category']=='ham', 'Category',]= 1


# In[21]:


X = data['Message']
Y = data['Category']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 3)


# In[25]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[26]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[30]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[31]:


print(X_train)


# In[32]:


print(X_train_features)


# In[33]:


model = LogisticRegression()


# In[34]:


model.fit(X_train_features, Y_train)


# In[35]:


prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[36]:


print('Acc on training data : ', accuracy_on_training_data)


# In[38]:


prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[39]:


print('Acc on test data : ', accuracy_on_test_data)


# In[45]:


input_your_mail = ["Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"

]

input_data_features = feature_extraction.transform(input_your_mail)

prediction = model.predict(input_data_features)
print(prediction)
if(prediction[0]==1):
    print('It is a Ham mail')
else:
    print('It is a Spam mail')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




