#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
import pandas as pd


# In[6]:


data = pd.read_csv('dataset.csv')


# In[8]:


x = data['Head'].tolist()
y = data['Brain'].tolist()


# In[9]:


x = sm.add_constant(x)


# In[10]:


result = sm.OLS(y, x).fit()


# In[11]:


print(result.summary())


# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[13]:


data = pd.read_csv('dataset.csv')


# In[14]:


x = data['Head'].tolist() 
y = data['Brain'].tolist() 
plt.scatter(x, y) 


# In[15]:


max_x = data['Head'].max() 
min_x = data['Head'].min() 


# In[17]:


x = np.arange(min_x, max_x, 1)


# In[25]:


y = 325.5734 * x + 0.2634 #final equation of regression


# In[ ]:




