
<div align="center">
  <img src="https://user-images.githubusercontent.com/43999912/122243751-fcd96080-cee1-11eb-9b88-4c334aa78e58.png">
</div>

# Introduction

A light weight wrapper to ease the deep learning developement workflow. Popnet consists of a collection of CNN architectures which are to be used instantly making object classiication easy and quick. Currently we have implemented AlexNET, ResNETs, MobileNets , VGG-16, Inception networks and some basic CNNs.  

# Installation

```sh
git clone https://github.com/nakul-shahdadpuri/popnet.git
```
Into the directory to be used in.
Soon the library will be available via PiP

# Using the Library

Popnet offers functions to load a dataset and make models and train them also test them in under 10 lines of code. This library is targetted towards people who just want to make a model without going into the theory behind it. 

## Training and Model Selection

```py
#imports
import popnet

#parameters
Batch = 32
Dataset = 'dataset'
Image_size = (200,200)
Tensor_size = (200,200,3)
Split = 0.3
epochs = 2
Name = "popnet"

#Actual Code begins here
training, testing = popnet.get_dataset_from_dir(Batch,Image_size,Dataset,Split)
classes = popnet.classes(training)
model = popnet.cnn_model(len(classes),Tensor_size)
model = popnet.train_model(model,training,testing,epochs)
# and done !! Time to Train

#Time to save the model to disk
popnet.save_model(model,Name)

```

## Testing and Accuracy Analysis

```py
#imports
import popnet

#Parameters
Batch = 32
Dataset = 'dataset'
Image_size = (200,200)
Split = 0.3

#Getting Dataset
model = popnet.load_model("model")

#Testing
training, testing = popnet.get_dataset_from_dir(Batch,Image_size,Dataset,Split)
popnet.test_model(model,testing)
```

## Dependencies

```sh
tensorflow==2.5.0
```