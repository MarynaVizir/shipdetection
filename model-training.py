import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from matplotlib.image import imread
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping


'''Creating variables for reading the data from folder (MacOS syntax)'''
data_dir = '../airbus-ship-detection'
train_path = data_dir + '/train_v2/'
df = pd.read_csv(data_dir + '/train_ship_segmentations_v2.csv')


'''Image-data processing'''
x_names = df['ImageId'].values      # collecting names
x = np.array([imread(train_path + n) for n in x_names])/255      # collecting images as pixel arrays, and normalization


'''Label-data processing'''
y = df['EncodedPixels'].notnull().apply(int).values    # binary mode as a result of ship recognition
y_cat = to_categorical(y)           # transforming to 2-categorical array


'''Creating random sets for training (70%) and testing (30%)'''
X_train, X_test, y_train, y_test = train_test_split(x, y_cat, test_size=0.3, random_state=37)


'''Creating a model instance'''
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


'''Training the Model'''
batch_size = 16
image_shape = X_train[0].shape
early_stop = EarlyStopping(monitor='val_loss', patience=2)

model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test),
          callbacks=[early_stop])
model.save('ship_detection.h5')            # saving the entire model to HDF5 file
