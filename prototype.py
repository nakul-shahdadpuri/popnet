#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports


# In[37]:


import tensorflow as tf
import os
import  matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation


# In[38]:


# making the dataset


# In[39]:


BATCH_SIZE = 32
IMG_SIZE = (200, 200)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=69)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='validation',
                                             seed=69)


# In[40]:


for image_batch,labels in train_dataset:
    print(image_batch.shape)
    print(labels.shape)
    break


# In[41]:


# now since the dataset has been imported we can create a model to implement it
# lets convert data to grey scale


# In[42]:


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# In[43]:



def make_model(num_of_classes,input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    x = tfl.Conv2D(15,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(input_img)
    x = tfl.ReLU()(x) 
    x = tfl.MaxPool2D(8,padding="SAME")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
 
     
    x = tfl.Conv2D(30,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
    x = tfl.ReLU()(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
  
    x = tfl.Conv2D(45,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(input_img)
    x = tfl.ReLU()(x) 
    x = tfl.MaxPool2D(4,padding="SAME")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)

    x = tfl.Conv2D(60,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
    x = tfl.ReLU()(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
    
    x = tfl.Conv2D(60,3,padding='SAME',activation="linear",input_shape=input_shape[1:])(x)
    x = tfl.ReLU()(x)
    x = tfl.MaxPool2D(4,padding="SAME")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
    
    x = tfl.Flatten()(x)
    
    x = tfl.Dense(200, activation="relu")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
    
    x = tfl.Dense(100, activation="relu")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)

    x = tfl.Dense(50, activation="relu")(x)
    x = tfl.Dropout(0.2)(x)
    x = tfl.BatchNormalization()(x)
    
    outputs = tfl.Dense(num_of_classes,activation="softmax")(x)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model


# In[44]:


num_of_classes = 10
img_size = (200,200,3)
model = make_model(num_of_classes,img_size)


# In[45]:


model.summary()


# In[46]:


model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])


# In[ ]:

print("Model has started Training ")
print(model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=10
))
print("model has finished training")


# In[ ]:


model.save("animals_model")

