
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_ac4644f90d0d458f8f10a1bd7a67824f = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='BNR2KuVMbcNxWkS8PfJxrlh1-KqWkQpA_C5dz6hdTRll',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_ac4644f90d0d458f8f10a1bd7a67824f.get_object(Bucket='survivalanalysisofchronickidneydi-donotdelete-pr-dj7sc14dp1pqhl',Key='kidney_disease-1.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset= pd.read_csv(body)
dataset.head()



# In[3]:


dataset


# In[4]:


dataset.drop(['Unnamed: 22'],axis=1,inplace=True)
dataset.drop(['id'],axis=1,inplace=True)
dataset.drop(['pc','sg','bu','dm'],axis=1,inplace=True)
dataset.drop(['cad'],axis=1,inplace=True)


# In[5]:


dataset


# In[6]:


dataset.isnull().any()


# In[7]:


dataset['age'].fillna((dataset['age'].mean()),inplace=True) 
dataset['bp'].fillna((dataset['bp'].mean()),inplace=True)
dataset['al'].fillna((dataset['al'].mode().iloc[0]),inplace=True) 
dataset['su'].fillna((dataset['su'].mean()),inplace=True) 
dataset['rbc'].fillna((dataset['rbc'].mode().iloc[0]),inplace=True) 
dataset['bgr'].fillna((dataset['bgr'].mean()),inplace=True) 
dataset['sc'].fillna((dataset['sc'].mean()),inplace=True) 
dataset['hemo'].fillna((dataset['hemo'].mean()),inplace=True) 
dataset['pcv'].fillna((dataset['pcv'].mode().iloc[0]),inplace=True) 
dataset['wc'].fillna((dataset['wc'].mode().iloc[0]),inplace=True)  
dataset['rc'].fillna((dataset['rc'].mode().iloc[0]),inplace=True) 
dataset['htn'].fillna((dataset['htn'].mode().iloc[0]),inplace=True) 
dataset['appet'].fillna((dataset['appet'].mode().iloc[0]),inplace=True) 
dataset['pe'].fillna((dataset['pe'].mode().iloc[0]),inplace=True) 
dataset['ane'].fillna((dataset['ane'].mode().iloc[0]),inplace=True) 


# In[8]:


dataset['pcv'].value_counts()


# In[9]:


z=dataset['pcv'].value_counts()


# In[10]:


dataset['appet'].value_counts()


# In[11]:


dataset['ane']


# In[12]:


dataset.isnull().any()


# In[13]:


dataset


# In[14]:


x=dataset.iloc[:,0:15].values


# In[15]:


p=x[:,8]
p


# In[16]:


p[66]=41


# In[17]:


p[214]=43


# In[18]:


p=x[:,8]
p


# In[19]:


p[76]=6200


# In[20]:


p[133]=8400


# In[21]:


p=x[:,9]
p


# In[22]:


p[214]=43


# In[23]:


p[185]=5800


# In[24]:


p[133]=8400


# In[25]:


p=x[:,10]
p


# In[26]:


p[162]=4.8


# In[27]:


p=x[:,1]
p


# In[28]:



y=dataset.iloc[:,-1].values


# In[29]:


p[133]=8400


# In[30]:


y


# In[31]:


y[230]='ckd'


# In[32]:


y[37]='ckd'


# In[33]:


y


# In[34]:


x[0]


# In[35]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[36]:


x[:,4]=lb.fit_transform(x[:,4])
x[:,11]=lb.fit_transform(x[:,11])
x[:,12]=lb.fit_transform(x[:,12])
x[:,13]=lb.fit_transform(x[:,13])
x[:,14]=lb.fit_transform(x[:,14])


# In[37]:


x.shape


# In[38]:


x[0]


# In[39]:


y=lb.fit_transform(y)


# In[40]:


y


# In[41]:


x[0]


# In[42]:


x.shape


# In[43]:


lb_y=LabelEncoder()
y=lb_y.fit_transform(y)
y


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# # Random Forest

# In[46]:


from sklearn.ensemble import RandomForestClassifier
RFclassifier=RandomForestClassifier(n_estimators=30,criterion='gini',random_state=0)


# In[47]:


RFclassifier.fit(x_train,y_train)


# In[48]:


y_RFpredict=RFclassifier.predict(x_test)


# In[49]:


y_RFpredict


# In[50]:


y_test


# In[51]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_RFpredict)


# # confution matrix

# In[52]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_RFpredict)
cm


# In[53]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_RFpredict)
roc_auc1=metrics.auc(fpr,tpr)
roc_auc1


# # Decision Tree

# In[54]:


from sklearn.tree import DecisionTreeClassifier
DTclassifier=DecisionTreeClassifier(criterion='entropy',random_state=0) #inplaces of entropy we can change name into "gini" so that we can calculate another method for decision


# In[55]:


DTclassifier.fit(x_train,y_train)


# In[56]:


DTclassifier.fit(x_train,y_train)


# In[57]:


y_DTpredict=DTclassifier.predict(x_test)


# In[58]:


y_DTpredict


# In[59]:


y_test


# In[60]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_DTpredict)


# In[61]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_DTpredict)
roc_auc2=metrics.auc(fpr,tpr)
roc_auc2


# # K N Neighbor

# In[62]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[63]:


classifier.fit(x_train,y_train)


# In[64]:


y_predict=classifier.predict(x_test)


# In[65]:


y_predict


# In[66]:


y_test


# In[67]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# In[68]:


import sklearn.metrics as metrics
fpr,tpr,threshold=metrics.roc_curve(y_test,y_predict)
roc_auc3=metrics.auc(fpr,tpr)
roc_auc3


# In[69]:


x=["rf","dt","knn"]


# In[70]:


y=[roc_auc1,roc_auc2,roc_auc3]


# In[71]:


plt.bar(x,y,width=0.6)
plt.title("multibars")
plt.show()


# In[72]:


y


# In[73]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[74]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[75]:


wml_credentials={
  "instance_id": "d346d3fb-4b13-40f7-ab38-44df3aa5df0d",
  "password": "07d6737a-ae37-49f4-8d12-412611fc8aac",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "13e55c54-3ec0-4b64-aa9a-ad8138c96cdc"
}


# In[76]:


client = WatsonMachineLearningAPIClient(wml_credentials)


# In[77]:


import json
instance_detail = client.service_instance.get_details()
print(json.dumps(instance_detail,indent=2
                ))


# In[78]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME:"SAIKRISHNA KUSUMA",
               client.repository.ModelMetaNames.AUTHOR_EMAIL:"saikrishna.ravinder@gmail.com",
               client.repository.ModelMetaNames.NAME:"Chronic Kidney Diseases"}


# In[79]:


model_artifact =client.repository.store_model(classifier,meta_props=model_props)


# In[80]:


publish_model_uid = client.repository.get_model_uid(model_artifact)


# In[83]:


publish_model_uid


# In[89]:


client.deployments.list()


# In[91]:


client.deployments.delete("3f41fb0d-09eb-4f2c-836f-ba4a1d099023")


# In[96]:


client.deployments.delete("32fe2922-644a-41f7-96db-82d5a0b4b9e3")


# In[97]:


created_deployment = client.deployments.create(published_model_uid, name="survival analysis of chronic kidney disease")


# In[101]:


scoring_endpoint = client.deployments.get_scoring_url(created_deployment)


# In[105]:


scoring_endpoint


# In[104]:


client.deployments.list()

