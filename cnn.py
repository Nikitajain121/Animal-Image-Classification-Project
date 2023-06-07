# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g_c-kJYh2vvxB9MOGcXnM3wkET-A8c_P
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt

translations = ['cavallo','elefante','gallina','cane','farfalla']
X = []
Y = []
for i in range(5):
    path = '/content/drive/MyDrive/animal_10/raw-img/' + translations[i]
    
    for file in os.listdir(path):
        try: 
            imgArray = cv2.imread(os.path.join(path, file))
            
#             resize images so they are all square and consistent
            imgArray = cv2.resize(imgArray, (100,100))
    
            X.append(imgArray)
            Y.append(i)
        except:
            pass

X = np.array(X)
Y = np.array(Y)
print('X shape: ', X.shape)
print('Y shape: ', Y.shape)

xtrain,xtest, ytrain,ytest=train_test_split(X,Y,test_size=0.1, random_state=42)

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)



print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

input_shape = xtrain[0].shape
num_classes = 5

import tensorflow as tf

tf.keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation = 'relu'))

model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(units=num_classes, activation = 'softmax'))

#model.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

history = model.fit(x = xtrain, 
    y = ytrain,
    epochs = 15,
    batch_size = 1000,
    validation_split=0.1,
    verbose=1)

model.save('mymodel.pkl')

model.save('my_model.h5')

from keras.models import save_model
save_model(model, "model.h5")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

history = pd.DataFrame(history.history)
display(history)

import pickle
pickle.dump(model, open('model.pkl','wb'))

label_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken']

pred=model.predict("OIP-_3S-iEDMQnko7ZHgq_FTcwHaEL.jpeg")

predictions = model.predict(xtest)

predictions = np.argmax(predictions, axis = 1)

print(classification_report(ytest, predictions, target_names = label_names))
    
cm = confusion_matrix(ytest, predictions)

plt.figure(figsize=(7,10))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=label_names, yticklabels=label_names)
plt.ylabel('Actual value')
plt.xlabel('Predicted value')
plt.title('Confusion Matrix', size = 15)

predictions

xtest

predictions = model.predict(xtest)

predictions

import numpy as np
from PIL import Image

img = Image.open('/content/drive/MyDrive/animal_10/raw-img/cane/OIP-_3S-iEDMQnko7ZHgq_FTcwHaEL.jpeg')
img = img.resize((100, 100))  # resize your image to the input shape of your CNN model
img_array = np.array(img)  # convert your image to a numpy array
img_array = np.expand_dims(img_array, axis=0)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

img_array = preprocess_input(img_array)  # apply the preprocessing function specific to your CNN model

pred = model.predict(img_array)

print(pred)

class_idx = np.argmax(pred, axis=1)[0]

class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken']

class_name = class_names[class_idx]

print('The predicted class is:', class_name)