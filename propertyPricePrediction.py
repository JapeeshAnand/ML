#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("housingData.csv")


# In[3]:


#housing.info()


# In[4]:


#housing.describe()


# In[5]:


matplotlib inline


# In[6]:


#import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20,15))


# In[7]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[9]:


housing = strat_train_set.drop(['MEDV'], axis=1)
housing_labels = strat_train_set['MEDV'].copy()


# In[10]:


housing['TAXRM'] = housing['TAX']/housing['RM']


# In[11]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing)


# In[12]:


X = imputer.transform(housing)


# In[13]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[14]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    (('imputer'), SimpleImputer(strategy='median')),
    (('std_scaler'), StandardScaler())
])


# In[15]:


housing_num = my_pipeline.fit_transform(housing)


# In[16]:


housing_num.shape


# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num, housing_labels)


# In[18]:


data = housing.iloc[:5]
labels = housing_labels.iloc[:5]


# In[19]:


prepared_data= my_pipeline.transform(data)


# In[20]:


model.predict(prepared_data)


# In[21]:


from sklearn.metrics import mean_squared_error
import numpy as np
housing_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[22]:


rmse


# In[23]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[24]:


rmse_scores


# In[25]:


from joblib import dump, load
dump(model, 'propertyPredition.joblib')


# In[26]:


X_test_prepared = my_pipeline.transform(housing)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(housing_labels, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




