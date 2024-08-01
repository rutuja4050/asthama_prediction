#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[26]:


data=pd.read_csv("asthma_disease_data.csv")
data


# In[27]:


data.head()


# In[28]:


data.tail()


# In[29]:


data.info()


# In[30]:



data.describe()


# In[31]:



data.columns


# In[32]:


data.shape


# In[33]:


data=data.drop(columns='PatientID', axis=1)
data


# In[34]:


data=data.drop(columns='DoctorInCharge', axis=1)
data


# In[40]:


#checking the distribution of target variables
data['Diagnosis'].value_counts()


# In[41]:


#Applying Smote to counter Data imbalance issues
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load your asthma dataset
# data = pd.read_csv("asthma_disease_data.csv")

# Split the dataset into features (X) and target (y)
X = data.drop('Diagnosis', axis=1)  # Replace 'target_column' with the actual name of your target column
y = data['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the original distribution
print("Before Over Sampling, count of the label '1': {}".format(sum(y_train == 1)))  
print("Before Over Sampling, count of the label '0': {} \n".format(sum(y_train == 0)))  

# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Check the new distribution
print('After Over Sampling, the shape of the train_X: {}'.format(X_train_res.shape))  
print('After Over Sampling, the shape of the train_y: {} \n'.format(y_train_res.shape))  
print("After Over Sampling, count of the label '1': {}".format(sum(y_train_res == 1)))  
print("After Over Sampling, count of the label '0': {}".format(sum(y_train_res == 0)))  


# In[43]:





# In[39]:


#splitting the features and target
#X=data.drop(columns='Diagnosis', axis=1)
#Y=data['Diagnosis']
#print(X)


# In[13]:


#print(Y)


# In[44]:


data['BMI']=data['BMI'].round()
data


# In[45]:


data['PhysicalActivity']=data['PhysicalActivity'].round()
data


# In[46]:


data['DietQuality']=data['DietQuality'].round()
data


# In[47]:


data['SleepQuality']=data['SleepQuality'].round()
data


# In[48]:


data['PollutionExposure']=data['PollutionExposure'].round()
data


# In[49]:


data['LungFunctionFEV1']=data['LungFunctionFEV1'].round()
data


# In[50]:


data['LungFunctionFVC']=data['LungFunctionFVC'].round()
data


# In[51]:


data.head()


# In[52]:


data.tail()


# In[23]:


#calling train test split
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[24]:


print(X.shape,X_train.shape,X_test.shape)


# In[54]:


#Model Training with Logistic regression
model=LogisticRegression()


# In[55]:


model.fit(X_train_res,y_train_res)
LogisticRegression()


# In[56]:


#Accuracy on training data
X_train_prediction=model.predict(X_train_res)
training_data_accuracy=accuracy_score(X_train_prediction,y_train_res)
print('Accuracy on training data is :',training_data_accuracy)


# In[59]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print('Accuracy on test data is :',test_data_accuracy)


# In[29]:


# Buidling a predictive system : 
input_data = (26,1,2,2,22.75704209,0,5.897329494,6.341014021,5.15396637,1.969838336,7.457664778,6.58463121,0,0,1,0,0,0,2.197767332,1.702393427,1,0,0,1,1,1	)

#Changing the input data copied from the excel file to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[30]:


#Reshape as we are predicting for only one instance:
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[31]:


prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print('This person has a healthy Lung functioning')
else:
    print('The person may have Asthma diseases issues')


# In[32]:





# In[33]:


#Model Training with Logistic regression
model=LogisticRegression()


# In[34]:


model.fit(X_train,Y_train)
LogisticRegression()


# In[61]:


#Accuracy on training data
X_train_prediction=model.predict(X_train_res)
training_data_accuracy=accuracy_score(X_train_prediction,y_train_res)
print('Accuracy on training data is :',training_data_accuracy)


# In[63]:


X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,y_test)
print('Accuracy on test data is :',test_data_accuracy)


# In[70]:


# Buidling a predictive system : 
#input_data = (5035,26,1,2,2,22.75704209,0,5.897329494,6.341014021,5.15396637,1.969838336,7.457664778,6.58463121,0,0,1,0,0,0,2.197767332,1.702393427,1,0,0,1,1,1	)
#input_data = (7422,18,1,0,1,20.74084988,0,5.805180182,4.386992164,7.731192173,7.733982865,2.279072823,6.467701123,0,0,0,1,0,0,1.132977278,5.509502035,0,0,0,1,1,0)
input_data = (7418,31,0,0,2,31.82,1,8.516835222,3.532327646,9.442670028,9.240483246,8.298978642,2.287043459,0,1,0,0,0,1,1.883568104,2.592148096,0,0,0,0,1,1)
#input_data=(46,1,0,2,2.0,0,9.0,7.0,7.0,1.0,70,5.0,0,1,1,0,0,1,3.0,2.0,0,1,1,0,1,1)
#input_data = (46, 1, 0, 2, 2.0, 0, 9.0, 7.0, 7.0, 1.0, 70, 5.0, 0, 1, 1, 0, 0, 1, 3.0, 2.0, 0, 1, 1, 0, 1, 1)
#Changing the input data copied from the excel file to numpy array
input_data_as_numpy_array = np.asarray(input_data)


# In[71]:


#Reshape as we are predicting for only one instance:
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[72]:


prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print('This person has a healthy Lung functioning')
else:
    print('The person may have Asthma diseases issues')


# In[ ]:




