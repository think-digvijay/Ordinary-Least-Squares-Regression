#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('dataset.csv')
print(dataset.shape)
dataset.head()


# In[3]:


X = dataset['Head'].values
Y = dataset['Brain'].values

x_mean = np.mean(X)
y_mean = np.mean(Y)

n = len(X)


# In[4]:


# calculating regression coeeficients

numerator = 0
denominator = 0

for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2

b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

print(b1, b0)


# In[5]:


# plotting the garphs

x_max = np.max(X) + 100
x_min = np.min(X) - 100

x = np.linspace(x_min, x_max, 1000)
y = b0 + b1 * x

plt.plot(x, y, color = "#00ff00", label='Linear Regression')
plt.scatter(X, Y, color = "#ff0000", label = 'Data point')

plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight (grams)')

plt.legend()
plt.show()


# In[7]:


# measuring accuracy of the model

rmse = 0

for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
        
rmse = np.sqrt(rmse/n)
print(rmse)


# In[8]:


# finding R2 scores

sumofsquares = 0
sumofresiduals = 0
for i in range(n) :
    y_pred = b0 + b1 * X[i]
    sumofsquares += (Y[i] - y_mean) ** 2
    sumofresiduals += (Y[i] - y_pred) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[ ]:




