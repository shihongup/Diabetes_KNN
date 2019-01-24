
# coding: utf-8

# <h1><font size="6">CSCI 6364 - Machine Learning</font></h1>
# <h1><font size="5">Project 1 - Pima Indians Diabetes</font></h1>
# <p><font size="4"><span style="line-height:30px;">Student: Shifeng Yuan</span><br>
# <span style="line-height:30px;">GWid: G32115270<span><br>
# Language: Python<font><br>
# <span style="line-height:30px;">Resource: Pima Indians Diabetes from Kaggle <span></p>

# In[91]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns


# <h1>1. Dataset Details</h1>
# <p> &nbsp; <font size="3">The dataset includes data from 768 women with 8 characteristics, in particular:</font></p>
# <ol><font size="3">
# <li>Number of times pregnant</li>
# <li>Plasma glucose concentration a 2 hours in an oral glucose tolerance test</li>
# <li>Diastolic blood pressure (mm Hg)</li>
# <li>Triceps skin fold thickness (mm)</li>
# <li>2-Hour serum insulin (mu U/ml)</li>
# <li>Body mass index (weight in kg/(height in m)^2)</li>
# <li>Diabetes pedigree function</li>
# <li>Age (years)</li>
# <font></ol>

# <h2> Inspect the Dataset </h2>

# In[105]:


# Read the dataset and then print the head
dataset = pd.read_csv('data/diabetes.csv')
print( len(dataset) )
print( dataset.head() )


# <h2>Dataset Visualization</h2>

# In[93]:


import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20, 15))
plt.show()


# In[94]:


column_x = dataset.columns[0:len(dataset.columns) - 1]
corr = dataset[dataset.columns].corr()
sns.heatmap(corr, annot = True)


# <h2>Data Spliting</h2>
# <p><font size="3">Usually, we divide our dataset into 2 to 3 parts. Here, I split the dataset into training data (80%)  and testing data(20%)</font></p>

# In[95]:


#split dataset
# x is the columns
X = dataset.iloc[:, 0:8]
# y is the last column which is the result
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# <h1>2. Algorithm Description</h1>
# <p><font size="3">As we see in the dataset, there are some data of zeroes and null, and they will negatively influence the accuracy of our traning. In this case, I decide to replace them with the median value of the columns they locate.</font></p>

# In[96]:


# replace zeroes
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# For the columns that should not contain zeroes and null, I replace all the zeroes with NaN 
# and then change all NaN to the median of that column
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)
    


# <p><font size="3">We should also standardize the dataset, here, I use the sklearn.preprocessing.StandardScaler() to standardize the dataset.This package use a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1. Below is the format:</font></p>
# <img src="standardization.png" width="500" height="500">

# In[97]:


# Standardization
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# <h2>Selection of K</h2>
# <p><font size="3">The selection of value K is important for KNN, usually, we make K the square root of the size of the test sample.</font></p>

# In[98]:


import math
math.sqrt(len(y_test))


# <p><font size="3">The package sklearn.neighbors.KNeighborsClassifier implementing the K-nearest Neighbors
# classification.</font></p>
# <p><font size="3">In general, the k is better to be an odd number, so we make it 11.<br>
# Using the sklearn KNeighborsClassifier package, define the metric method as euclidean.
# This dataset is not too big, so we can just use a brute force algorithm.</font></p>

# In[99]:


#define the model: Init knn
classifier = KNeighborsClassifier(n_neighbors = 11, algorithm = 'brute', p = 2, metric = 'euclidean')


# <p><font size="3">Fit the model using X as training data and y as target values</font></p>

# In[100]:


#Fit model
classifier.fit(X_train, y_train)


# <h1>3. Algorithm Results</h1>
# <p><font size="3"></font></p>

# In[101]:


#predict the test result
y_pred = classifier.predict(X_test)


# In[102]:


#Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)


# <p><font size="3">The table below shows the true positive, true negative, false positive and false negatives. In total 153 test samples, we get 92 true positives and 27 true negatives, which makes the accuracy score be 0.7727272727272727.</font></p>
# <img src="diabetes92.png" width="500" height="500">

# In[103]:


print(accuracy_score(y_test, y_pred))


# <h1>4. Runtime</h1>
# <p><font size="3"></font></p>

# <p><font size="3">
#     For d dimension, we need O(d) runtime to compute one distance between two data, so computing all the distance between one data to other data needs O(nd) runtime, then we need O(kn) runtime to find the K nearest neibors, so, in total, it takes O(dn+kn) runtime for the classifier to classify the data.
# </font></p>

# In[106]:


import time
start = time.time()
classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, algorithm='brute', metric = 'euclidean')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
end = time.time()
print(end-start)


# <p><font size="3">
#     As is shown above, the "wall-clock" of the runtime is about 0.0318s
# </font></p>
