#!/usr/bin/env python
# coding: utf-8

# In[229]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[230]:


cc_apps=pd.read_table("C:/Users/Cemre/Desktop/cc_approvals.txt",sep=",", header=None, names=['Male', 'Age', 'Debt','Married', 'BankCustomer', 'EducationLevel','Ethnicity', 'YearsEmployed', 'PriorDefault','Employed', 'CreditScore', 'DriversLicense','Citizen', 'ZipCode', 'Income','Approved'])


# In[231]:


cc_apps


# In[232]:


cc_apps.info()


# In[233]:


cc_apps.describe()


# In[234]:


cc_apps.isnull().values.sum()


# In[235]:


cc_apps=cc_apps.replace("?",np.NaN)


# In[236]:


cc_apps.fillna(cc_apps.mean(), inplace=True)

print(cc_apps.isnull().sum())


# In[237]:


for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

print(cc_apps.isnull().sum())


# In[238]:


le=LabelEncoder()

for col in cc_apps.columns.values:
    if cc_apps[col].dtypes=='object':
        cc_apps[col]=le.fit_transform(cc_apps[col])


# In[239]:


cc_apps = cc_apps.drop([cc_apps.columns[10],cc_apps.columns[13]], axis=1)


# In[240]:


cc_apps = cc_apps.values

X,y = cc_apps[:,0:13] , cc_apps[:,13]

X_train, X_test, y_train, y_test = train_test_split(X
                                                    ,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)


# In[241]:


scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX_train = scaler.fit_transform(X_train)

rescaledX_test = scaler.transform(X_test)


# In[242]:


logreg = LogisticRegression()

logreg.fit(rescaledX_train,y_train)


# In[243]:


y_pred = logreg.predict(rescaledX_test)

print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test,y_test))

confusion_matrix(y_test,y_pred)


# In[244]:


tol = [0.01, 0.001 ,0.0001]

max_iter = [100, 150, 200]

param_grid = dict(tol=tol, max_iter=max_iter)


# In[245]:


grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

rescaledX = scaler.fit_transform(X)

grid_model_result = grid_model.fit(rescaledX, y)

best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_

print("Best: %f using %s" % (best_score, best_params))

