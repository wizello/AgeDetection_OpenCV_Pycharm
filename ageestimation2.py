

import os
import shutil



train_data_dir = 'data'
class_names = ['adult', 'child', 'teenager', 'elderly', 'baby']

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
import keras

#Creating the Model

base_model =  InceptionV3(input_shape=(256,256,3), include_top= False)

for layer in base_model.layers:
  layer.trainable = False

X = Flatten()(base_model.output)
X = Dense(units=5, activation= 'sigmoid')(X)

model = Model(base_model.input , X)
model.compile(optimizer= 'adam', loss = keras.losses.binary_crossentropy, metrics=['accuracy'])

model.summary()

#Data Augmentation and Preprocessing



train_datagen = ImageDataGenerator(featurewise_center= True ,
                                   rotation_range= 0.4,
                                   width_shift_range= 0.3,
                                   horizontal_flip= True,
                                   preprocessing_function= preprocess_input ,
                                   zoom_range= 0.4 ,
                                   shear_range= 0.4)
train_data = train_datagen.flow_from_directory(directory= 'data',
                                               target_size=(256,256),
                                               batch_size= 36)

train_data.class_indices

t_img, label = train_data.next()

t_img.shape

#Visualising the data

def plotImages(img_arr , label):
  for idx , img in enumerate( img_arr ):
    if idx <= 10 :
      plt.figure(figsize= (5,5))
      plt.imshow(img)
      plt.title(img.shape)
      plt.axis = False
      plt.show()

plotImages(t_img , label)

#Best Model

from keras.callbacks import ModelCheckpoint , EarlyStopping

mc = ModelCheckpoint(filepath= "./best_model.h5" ,
                     monitor='accuracy',
                     verbose= 1,
                     save_best_only= True )

es = EarlyStopping(monitor= "accuracy" ,
                   min_delta= 0.01,
                   patience= 5,
                   verbose= 1)

cb = [mc,es]

#Training the Model

his = model.fit_generator(train_data , steps_per_epoch= 10 , epochs= 30, callbacks= cb)

from keras.models import load_model
model = load_model("best_model.h5")

#Evaluating Training History

h = his.history
h.keys()

plt.plot(h['loss'], 'go--')
plt.plot(h['accuracy'] , 'go--' , c = "red" , )

plt.title("Loss vs Acc")
plt.show()

#Making Predictions


path = 'data/baby/baby2.jpg'
img = load_img (path, target_size=(256,256) )

i = img_to_array(img)

i = preprocess_input(i)

input_arr = np.array([i])
input_arr.shape

pred = np.argmax(model.predict(input_arr))

if pred == 0:
  print("The image is an adult")
elif pred == 1:
  print("The image is a baby")
elif pred == 2:
  print("The image is a child")
elif pred == 3:
  print("The image is a elderly")
else:
  print("The image is a teenager")

if pred == 0:
  age = "The persons age is between 20-4"
elif pred == 1:
  age = "The persons age is between 0-3"
elif pred == 2:
  age = "The persons age is between 4-12"
elif pred == 3:
  age = "The persons age is between 50-90"
else:
  age = "The persons age is between 13-19"

plt.imshow(input_arr[0])
plt.title(age)
plt.show()
