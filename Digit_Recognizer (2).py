#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(1212)

import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers


# In[2]:


data_test=pd.read_csv('test.csv')
data_train=pd.read_csv('train.csv')
df_test=pd.DataFrame(data_test)
df_train=pd.DataFrame(data_train)


# In[3]:


print(df_test.shape)
print(df_train.shape)


# In[4]:


df_features=df_train.iloc[:,1:785]
df_label=df_train.iloc[:,0]

X_test=df_test.iloc[:,0:784]
print(X_test.shape)
print(df_features.shape)


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv=train_test_split(df_features, df_label, test_size=0.2, random_state=1212)



X_train=X_train.as_matrix().reshape(33600, 784)

X_cv=X_cv.as_matrix().reshape(8400, 784)

X_test=X_test.as_matrix().reshape(28000, 784)


# In[6]:


print(min(X_train[1]), max(X_train[1]))


# In[7]:


#Feature Normalization

X_train=X_train.astype('float32')
X_cv=X_cv.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_cv/=255
X_test/=255

#Convert labels to one hot encoded
num=10
y_train=keras.utils.to_categorical(y_train, num)
y_cv=keras.utils.to_categorical(y_cv, num)


# In[8]:


#Printing 2 examples of label after conversion
print(y_train[0])
print(y_train[3])


# In[9]:


'''MODEL FITTING
    Input Parameters'''
n=784
n_hidden1=300
n_hidden2=100
n_hidden3=100
n_hidden4=200
num=10


# In[10]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)
out=Dense(num, activation='softmax', name="Output_Layer")(x)


# In[11]:


#Our model has 6 Layer: 1 Input Layer, 4 Hidden Layer and 1 Output Layer
model=Model(ip, out)
print(model.summary())


# In[12]:


#Parameters for the Neural Network
learning_rate=0.1
T_epochs=20
T_batch_size=100
sgd=optimizers.SGD(lr=learning_rate)


# In[13]:


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[14]:


#Training 1
T1=model.fit(X_train, y_train, 
             batch_size=T_batch_size,
            epochs=T_epochs,
            verbose=2,
            validation_data=(X_cv, y_cv))


# In[15]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)
out=Dense(num, activation='softmax', name="Output_Layer")(x)


# In[16]:


#We use Adam optimizer
adam=keras.optimizers.Adam(lr=learning_rate)
model2=Model(ip, out)
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[17]:


#training 2
T2=model2.fit(X_train, y_train,
             batch_size=T_batch_size,
             epochs=T_epochs,
             verbose=2,
             validation_data=(X_cv, y_cv))


# In[18]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)
out=Dense(num, activation='softmax', name="Output_Layer")(x)

learning_rate = 0.01


# In[19]:


#Model 2a
adam=keras.optimizers.Adam(lr=learning_rate)
model2a=Model(ip, out)
model2a.compile(loss='categorical_crossentropy',
                optimizer='adam',
               metrics=['accuracy'])


# In[20]:


model2a.fit(X_train, y_train,
           batch_size=T_batch_size,
           epochs=T_epochs,
           validation_data=(X_cv, y_cv),
           verbose = 2,)


# In[21]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)
out=Dense(num, activation='softmax', name="Output_Layer")(x)

learning_rate = 0.5


# In[22]:


#Model 2b
adam=keras.optimizers.Adam(lr=learning_rate)
model2b=Model(ip, out)
model2b.compile(loss='categorical_crossentropy',
                optimizer='adam',
               metrics=['accuracy'])


# In[23]:


model2b.fit(X_train, y_train,
           batch_size=T_batch_size,
           epochs=T_epochs,
           validation_data=(X_cv, y_cv))


# In[102]:


'''The accuracy, as measured by the 3 different learning rates 0.01, 0.1 and 0.5 are around 98%, 97% and 98% respectively. 
As there are no considerable gains by changing the learning rates, we stick with the default learning rate of 0.01.

We proceed to fit a neural network with 5 hidden layers with the features in the hidden layer set as (300, 100, 100, 100, 200) 
respectively. To ensure that the two models are comparable, we will set the training epochs as 20, 
and the training batch size as 100.'''


# In[103]:


#Parametes for Model3
n=784
n_hidden1=300
n_hidden2=100
n_hidden3=100
n_hidden4=100
n_hidden5=200
num=10


# In[104]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)
x=Dense(n_hidden5, activation='relu', name="Hidden_Layer_5")(x)
out=Dense(num, activation='softmax', name="Output_Layer")(x)


# In[105]:


model3=Model(ip,out)
model3.summary()


# In[106]:


adam=keras.optimizers.Adam(lr=0.01)

model3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[107]:


model3.fit(X_train, y_train,
           batch_size=T_batch_size,
          epochs=T_epochs,
          validation_data=(X_cv, y_cv))


# In[24]:


n=784
n_hidden1=300
n_hidden2=100
n_hidden3=100
n_hidden4=200
num=10


# In[25]:


'''Compared to our first model, adding an additional layer did not significantly improve the accuracy from our previous model. 
However, there are computational costs (in terms of complexity) in implementing an additional layer in our neural network. 
Given that the benefits of an additional layer are low while the costs are high, we will stick with the 4 layer neural network.

We now proceed to include dropout (dropout rate of 0.3) in our second model to prevent overfitting.'''


# In[26]:


ip=Input(shape=(784,)) #INPUT
x=Dense(n_hidden1, activation='relu', name="Hidden_Layer_1")(ip)
x = Dropout(0.3)(x)
x=Dense(n_hidden2, activation='relu', name="Hidden_Layer_2")(x)
x = Dropout(0.3)(x)
x=Dense(n_hidden3, activation='relu', name="Hidden_Layer_3")(x)
x = Dropout(0.3)(x)
x=Dense(n_hidden4, activation='relu', name="Hidden_Layer_4")(x)

out=Dense(num, activation='softmax', name="Output_Layer")(x)


# In[27]:


model4=Model(ip, out)
model4.summary()


# In[28]:


model4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[29]:


model4.fit(X_train, y_train,
          batch_size=T_batch_size,
          epochs=T_epochs,
          validation_data=(X_cv, y_cv))


# In[30]:


test_pred = pd.DataFrame(model4.predict(X_test, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.head()


# In[31]:


test_pred.to_csv('mnist_submission.csv', index = False)