#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the packages
#Data processing packages
import numpy as np 
import pandas as pd 

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

#Machine Learning packages
from sklearn.svm import SVC,NuSVC
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler	
from sklearn.metrics import confusion_matrix

#Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
df['Over18'].value_counts()
df.describe()
df = df.drop(['EmployeeCount','Over18'], axis = 1)
df['Attrition']=df['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)
df.head()
df=pd.get_dummies(df)
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Distribution of Attrition
plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', data=df, palette='Set1')
plt.title('Distribution of Attrition')
plt.show()

# Attrition by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender_Male', hue='Attrition', data=df, palette='Set2')
plt.title('Attrition by Gender')
plt.show()

# Attrition by Department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department_Sales', hue='Attrition', data=df, palette='Set3')
plt.title('Attrition by Department')
plt.show()

# Attrition by Age
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Age', hue='Attrition', multiple='stack', palette='Set1')
plt.title('Attrition by Age')
plt.show()

# Attrition by Monthly Income
plt.figure(figsize=(10, 6))
sns.histplot(df, x='MonthlyIncome', hue='Attrition', multiple='stack', palette='Set2', bins=30)
plt.title('Attrition by Monthly Income')
plt.show()

# Attrition by Job Role
plt.figure(figsize=(14, 8))
sns.countplot(y='JobRole_Research Scientist', hue='Attrition', data=df, palette='Set3')
plt.title('Attrition by Job Role')
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Attrition by Years at Company
plt.figure(figsize=(10, 6))
sns.histplot(df, x='YearsAtCompany', hue='Attrition', multiple='stack', palette='Set1', bins=15)
plt.title('Attrition by Years at Company')
plt.show()


# In[3]:


X = df.drop(['Attrition'], axis=1)
y=df['Attrition']


# In[4]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)


# In[12]:


def train_test_ml_model(X_train,y_train,X_test,Model):
    model.fit(X_train,y_train) #Train the Model
    y_pred = model.predict(X_test) #Use the Model for prediction

    # Test the Model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_pred)
    accuracy = round(100*np.trace(cm)/np.sum(cm),1)

    #Plot/Display the results
    cm_plot(cm,Model)
    print('Accuracy of the Model' ,Model, str(accuracy)+'%')
#Function to plot Confusion Matrix
def cm_plot(cm,Model):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Comparison of Prediction Result for '+ Model)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X = df.drop(columns=['Attrition'])
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'KNN': KNeighborsClassifier(),
    'XGBClassifier': XGBClassifier(),
    'NuSVC': NuSVC(),
    'SVC': SVC()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Attrition', 'Attrition'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()


# In[8]:


from sklearn.svm import SVC,NuSVC  #Import packages related to Model
Model = "SVC"
model=SVC() #Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


# In[9]:


from sklearn.svm import SVC,NuSVC  #Import packages related to Model
Model = "NuSVC"
model=NuSVC(nu=0.285)#Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


# In[10]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
import xgboost
from xgboost import XGBClassifier  #Import packages related to Model
Model = "XGBClassifier()"
model=XGBClassifier() #Create the Model

train_test_ml_model(X_train,y_train,X_test,Model)


# In[11]:


from sklearn.neighbors import KNeighborsClassifier  #Import packages related to Model
Model = "KNeighborsClassifier"
model=KNeighborsClassifier()

train_test_ml_model(X_train,y_train,X_test,Model)


# In[ ]:




