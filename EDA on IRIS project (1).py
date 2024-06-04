#!/usr/bin/env python
# coding: utf-8

# In[25]:


pwd


# In[1]:


import pandas as pd
df=pd.read_csv("Iris.csv")
df.head()


# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


data=df.drop_duplicates("Species")
data


# In[7]:


df.value_counts("Species")


# In[8]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.countplot(x='Species', data=df, ) 
plt.show()


# In[9]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', 
				hue='Species', data=df, ) 

# Placing Legend outside the Figure 
plt.legend(bbox_to_anchor=(1, 1), loc=2) 

plt.show()


# In[10]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', 
				hue='Species', data=df, ) 

# Placing Legend outside the Figure 
plt.legend(bbox_to_anchor=(1, 1), loc=2) 

plt.show()


# In[11]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.pairplot(df.drop(['Id'], axis = 1), 
			hue='Species', height=2)


# In[12]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


fig, axes = plt.subplots(2, 2, figsize=(10,10)) 

axes[0,0].set_title("Sepal Length") 
axes[0,0].hist(df['SepalLengthCm'], bins=7) 

axes[0,1].set_title("Sepal Width") 
axes[0,1].hist(df['SepalWidthCm'], bins=5); 

axes[1,0].set_title("Petal Length") 
axes[1,0].hist(df['PetalLengthCm'], bins=6); 

axes[1,1].set_title("Petal Width") 
axes[1,1].hist(df['PetalWidthCm'], bins=6);


# In[38]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "SepalLengthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "SepalWidthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "PetalLengthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "PetalWidthCm").add_legend() 

plt.show()


# In[14]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

def graph(y): 
	sns.boxplot(x="Species", y=y, data=df) 

plt.figure(figsize=(10,10)) 
	
# Adding the subplot at the specified 
# grid position 
plt.subplot(221) 
graph('SepalLengthCm') 

plt.subplot(222) 
graph('SepalWidthCm') 

plt.subplot(223) 
graph('PetalLengthCm') 

plt.subplot(224) 
graph('PetalWidthCm') 

plt.show()


# In[15]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Load the dataset 
df = pd.read_csv('Iris.csv') 

sns.boxplot(x='SepalWidthCm', data=df)


# In[23]:


# Importing 
import sklearn 
#from sklearn.datasets import load_boston 
import pandas as pd 
import seaborn as sns 
import numpy as np

# Load the dataset 
df = pd.read_csv('Iris.csv') 

# IQR 
Q1 = np.percentile(df['SepalWidthCm'], 25, 
				interpolation = 'midpoint') 

Q3 = np.percentile(df['SepalWidthCm'], 75, 
				interpolation = 'midpoint') 
IQR = Q3 - Q1 

print("Old Shape: ", df.shape) 

# Upper bound 
upper = np.where(df['SepalWidthCm'] >= (Q3+1.5*IQR)) 

# Lower bound 
lower = np.where(df['SepalWidthCm'] <= (Q1-1.5*IQR)) 

# Removing the Outliers 
df.drop(upper[0], inplace = True) 
df.drop(lower[0], inplace = True) 

print("New Shape: ", df.shape) 

sns.boxplot(x='SepalWidthCm', data=df)


# In[18]:


axis = df.plot.hist(bins=30, alpha=0.5)
axis.set_xlabel('Size in cm');


# In[20]:


sns.pairplot(df, hue='Species')


# In[22]:


sns.countplot(x = "Species", data = data)
plt.title('Species', fontsize = 20)
plt.rcParams["figure.figsize"]=15,10
plt.show()


# In[ ]:




