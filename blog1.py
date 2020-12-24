#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch

import torch.nn as nn
from torch.nn import BatchNorm1d, Dropout
from torch.nn import BCELoss
from torch.optim import Adam

from torch.nn import Linear, Sigmoid, Sequential,ReLU, Softmax
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# In[2]:


a= pd.read_csv(r"E:\assignment\insurance\train.csv")


# In[3]:


a.head()


# In[4]:


a["Response"].value_counts(),a["Response"].value_counts(1)


# In[5]:


data=a
data["Vehicle_Age"]=data["Vehicle_Age"].map({"< 1 Year": 1, "1-2 Year":2, "> 2 Years":3})
data["Gender"]=data["Gender"].map({"Male": 1, "Female":2})
data["Vehicle_Damage"]=data["Vehicle_Damage"].map({"Yes": 1, "No":2})
data['Gender']=data["Gender"].astype('float')
data["Response"]=data["Response"].astype('float')
data= data.drop("id", axis=1)
target= data["Response"]
data= data.drop("Response", axis=1)

for i in data.columns[1:]:
    data[i]= (data[i]- data[i].min())/(data[i].max()-data[i].min())

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target, random_state=42)


# In[6]:



class classification(Dataset):
    def __init__(self,path, label):
        self.x=path
        self.y =label
        
        
    def __getitem__(self, index):
        
        return self.x[index],self.y[index]
   
        
    def __len__(self):
        return len(self.y)
    
train_data= classification(torch.FloatTensor(x_train.values),torch.FloatTensor(y_train.values))
test_data= classification(torch.FloatTensor(x_test.values),torch.FloatTensor(y_test.values))
train_loader= DataLoader(train_data, batch_size=100)
test_loader= DataLoader(test_data, batch_size=100)


# In[7]:


for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
  break


# In[8]:


batch_X.shape, batch_y.shape


# In[9]:


model=Sequential(
            Linear(x_train.shape[1],100),
            Sigmoid(),
            # adding batch normalization layer
            BatchNorm1d(100),
            # adding dropout layer
            Dropout(0.5, inplace=True),
            Linear(100, 100),
    
            Sigmoid(),
            # adding batch normalization layer
            BatchNorm1d(100),
            # adding dropout layer
            Dropout(0.5, inplace=True),
            Linear(100, 1),
            Sigmoid())


# In[10]:


model


# In[11]:


optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCELoss()


# In[12]:


def train(net, train_loader):
    for epoch in range (10): # no. of epochs
        training_loss = 0
        acc=0
        for batch_idx, (batch_x, batch_y) in (enumerate(train_loader)):
            net.train()
            optimizer.zero_grad()
            outputs = net(batch_x)
            outputs= outputs.squeeze()
            loss = criterion(outputs, batch_y)
            output= outputs.detach().numpy()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
        training_loss = np.average(training_loss)
        acc1=0
        for i in range(len(output)):
                acc =(roc_auc_score(batch_y,output))
                acc1=acc
        acc= np.average(acc1)
    
        print( epoch, training_loss,acc)


# In[13]:


train(model, train_loader)


# In[14]:


test_prediction = []
test_target = []

for batch_idx, (batch_x, batch_y) in (enumerate(train_loader)):
    model.eval()
                # data pixels and labels to GPU if available
                ##inputs, labels = data[0], data[1]
                # set the parameter gradients to zero
    output = model(batch_x)
    output= output.squeeze()
    
    test_predicted_label = []
    for i in output:
        # defining the threshold value as 0.5
        if i>=.5:
            test_predicted_label.append(1)
        else:
            test_predicted_label.append(0)
    test_prediction.append(test_predicted_label)
    test_target.append(batch_y)    
            
            


# In[15]:


test_accuracy = []

for i in range(len(test_prediction)):
    test_accuracy.append(roc_auc_score(test_target[i].data.cpu(),test_prediction[i]))
    
print('training accuracy: \t', np.average(test_accuracy))


# In[16]:


test_prediction = []
test_target = []

for batch_idx, (batch_x, batch_y) in (enumerate(train_loader)):
    model.eval()
                # data pixels and labels to GPU if available
                ##inputs, labels = data[0], data[1]
                # set the parameter gradients to zero
    output = model(batch_x)
    output= output.squeeze()
    
    test_predicted_label = []
    for i in output:
        # defining the threshold value as 0.5
        if i>0.:
            test_predicted_label.append(1)
        else:
            test_predicted_label.append(0)
    test_prediction.append(test_predicted_label)
    test_target.append(batch_y)    
            
            


# In[17]:


test_accuracy = [] 
for i in range(len(test_prediction)):
    test_accuracy.append(roc_auc_score(test_target[i].data.cpu(),test_prediction[i]))

    
print('training accuracy: \t', np.average(test_accuracy))


# # weighed Random Sampler

# In[18]:


labels = np.array(y_train)
labels= labels.astype(int)
class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
num_samples= sum(class_sample_count)
weight = 1. / class_sample_count
class_weights= weight[labels]

sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(class_weights), int(num_samples))


# In[19]:


from torch.utils.data import DataLoader
from torch.utils.data import Dataset
class classification(Dataset):
    def __init__(self,path, label):
        self.x=path
        self.y =label
        
        
    def __getitem__(self, index):
        
        return self.x[index],self.y[index]
   
        
    def __len__(self):
        return len(self.y)
    
train_data= classification(torch.FloatTensor(x_train.values),torch.FloatTensor(y_train.values))
test_data= classification(torch.FloatTensor(x_test.values),torch.FloatTensor(y_test.values))
train_loader= DataLoader(train_data, batch_size=100, sampler=sampler)
test_loader= DataLoader(test_data, batch_size=100, sampler= sampler)


# In[20]:


def train1(net, train_loader):
    for epoch in range (10): # no. of epochs
        training_loss = 0
        acc=0
        for batch_idx, (batch_x, batch_y) in (enumerate(train_loader)):
            net.train()
            optimizer.zero_grad()
            outputs = net(batch_x)
            outputs= outputs.squeeze()
            loss = criterion(outputs, batch_y)
            output= outputs.detach().numpy()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            
        training_loss = np.average(training_loss)
        acc1=0
        for i in range(len(output)):
                acc =(roc_auc_score(batch_y,output))
                acc1=acc
        acc= np.average(acc1)
    
        print( epoch, training_loss,acc)


# In[21]:


train1(model, train_loader)


# In[22]:


test_prediction = []
test_target = []

for batch_idx, (batch_x, batch_y) in (enumerate(train_loader)):
    model.eval()
                # data pixels and labels to GPU if available
                ##inputs, labels = data[0], data[1]
                # set the parameter gradients to zero
    output = model(batch_x)
    output= output.squeeze()
    
    test_predicted_label = []
    for i in output:
        # defining the threshold value as 0.5
        if i>=.5:
            test_predicted_label.append(1)
        else:
            test_predicted_label.append(0)
    test_prediction.append(test_predicted_label)
    test_target.append(batch_y)    
            


# In[23]:


test_accuracy = []
for i in range(len(test_prediction)):
     test_accuracy.append(roc_auc_score(test_target[i].data.cpu(),test_prediction[i]))

print('training accuracy: \t', np.average(test_accuracy))


# In[24]:


c= pd.read_csv(r"E:\assignment\insurance\test.csv")


# In[25]:


c.head()


# In[26]:


data1=c
data1["Vehicle_Age"]=data1["Vehicle_Age"].map({"< 1 Year": 1, "1-2 Year":2, "> 2 Years":3})
data1["Gender"]=data1["Gender"].map({"Male": 1, "Female":2})
data1["Vehicle_Damage"]=data1["Vehicle_Damage"].map({"Yes": 1, "No":2})
data1['Gender']=data1["Gender"].astype('float')
data1= data1.drop("id", axis=1)
for i in data1.columns[1:]:
        data1[i]= (data1[i]- data1[i].min())/(data1[i].max()-data1[i].min())


# In[27]:


test_prediction=[]
data1= torch.FloatTensor(data1.values)
y_pred= model(data1)
y_pred= y_pred.squeeze()
    
test_predicted = []
for i in y_pred:
    
        # defining the threshold value as 0.5
    if i>=.5:
        test_predicted.append(1)
    else:
        test_predicted.append(0)
test_prediction.append(test_predicted)


# In[28]:


test_submit= pd.DataFrame(test_prediction)
test_submit.head()


# In[29]:


test_sub= test_submit.transpose()
test_sub.head()


# In[30]:


test_sub_nn= pd.concat([c, test_sub], axis=1)


# In[31]:


test_sub_final= test_sub_nn[["id",0]]


# In[32]:


test_sub_final.head()


# In[33]:


test_sub_final.to_csv("nn_weighed_model.csv")


# In[ ]:




