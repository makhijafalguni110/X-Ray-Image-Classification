import numpy
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
from keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import  load_img
import warnings
warnings.filterwarnings('ignore')

img_width, img_height = 224, 224
batchsize = 32
num_of_class = 2

train = keras. utils.image_dataset_from_directory(
    directory='/kaggle/input/chest-xray-pneumonia/chest_xray/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

validation = keras. utils.image_dataset_from_directory(
    directory='/kaggle/input/chest-xray-pneumonia/chest_xray/val',
    labels='inferred',
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

test = keras. utils.image_dataset_from_directory(
    directory='/kaggle/input/chest-xray-pneumonia/chest_xray/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batchsize,
    image_size=(img_width, img_height))

plt.pie([len(train), len(validation), len(test)],
        labels=['train', 'validation', 'test'], autopct='%.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.show()

print(train.class_names)
print(validation.class_names)
print(test.class_names)

import matplotlib.pyplot as plt

# Function to get a batch of sample images and labels from a dataset
def get_sample(dataset, num_samples=10):
    images = []
    labels = []
    for image_batch, label_batch in dataset.take(1):  # Extract one batch
        for i in range(min(num_samples, len(image_batch))):
            images.append(image_batch[i].numpy())
            labels.append(label_batch[i].numpy())
    return images, labels

# Get sample images and labels from the training dataset
sample_images_train, sample_labels_train = get_sample(train)

# Get sample images and labels from the validation dataset
sample_images_val, sample_labels_val = get_sample(validation)

# Get sample images and labels from the test dataset
sample_images_test, sample_labels_test = get_sample(test)

# Function to display sample images with labels
def show_sample_images(images, labels, class_names):
    plt.figure(figsize=(15, 15))
    for i in range(len(images)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].astype("uint8"))
        label = labels[i]
        plt.title(class_names[label.argmax()])
        plt.axis("off")

# Display sample images with labels for training dataset
show_sample_images(sample_images_train, sample_labels_train, class_names=["Normal", "Pneumonia"])
plt.suptitle("Sample Images with Labels (Training)")
plt.show()

# Display sample images with labels for validation dataset
show_sample_images(sample_images_val, sample_labels_val, class_names=["Normal", "Pneumonia"])
plt.suptitle("Sample Images with Labels (Validation)")
plt.show()
# Display sample images with labels for test dataset
show_sample_images(sample_images_test, sample_labels_test, class_names=["Normal", "Pneumonia"])
plt.suptitle("Sample Images with Labels (Test)")
plt.show()

# Extracting Features and Labels
x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature.numpy())
    y_train.append(label.numpy())

for feature, label in test:
    x_test.append(feature.numpy())
    y_test.append(label.numpy())
    
for feature, label in validation:
    x_val.append(feature.numpy())
    y_val.append(label.numpy())

# Concatenate the lists to get the full 'x' and 'y' arrays
x_train = np.concatenate(x_train, axis=0)
x_val = np.concatenate(x_val, axis=0)
x_test = np.concatenate(x_test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_val = np.concatenate(y_val, axis=0)
y_test = np.concatenate(y_test, axis=0)
# check the shapes of 'x_train' and 'y_train':
print("Shape of 'x_train':", x_train.shape)
print("Shape of 'y_train':", y_train.shape)
print("Shape of 'x_val':", x_val.shape)
print("Shape of 'y_val':", y_val.shape)
print("Shape of 'x_test':", x_test.shape)
print("Shape of 'y_test':", y_test.shape)

# Pixel Value Scaling for Datasets: Normalizing and Standardizing the Data
x_train=x_train/255
x_val=x_val/255
x_test=x_test/255

model=Sequential()

model.add(Conv2D(16, kernel_size=(3,3),activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.summary()
keras.utils.plot_model(model, show_shapes=True) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs= 25, validation_data=(x_val, y_val))

from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)

# Convert one-hot encoded labels back to categorical labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Define labels for the confusion matrix
labels = ['Normal', 'Pneumonia']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Extract the training and validation loss values from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a list of epoch numbers (1 to number of epochs)
epochs = range(1, len(train_accuracy) + 1)

# Plot the loss graph
plt.plot(epochs, train_accuracy , label='Training Acc')
plt.plot(epochs, val_accuracy, label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.grid(True)
plt.show()

model.save("cnn_v3.h5")

model=Sequential()

model.add(Conv2D(64, kernel_size=(3,3),activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.summary()
keras.utils.plot_model(model, show_shapes=True) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs= 25, validation_data=(x_val, y_val))

y_pred = model.predict(x_test)

# Convert one-hot encoded labels back to categorical labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Define labels for the confusion matrix
labels = ['Normal', 'Pneumonia']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(img_width, img_height, 3))
# Freeze the base model
for layer in conv_base.layers:
    layer.trainable = False

for i in range(3):
    conv_base.layers[-2-i].trainable = True

for layer in conv_base.layers:
    print(layer.name,layer.trainable)

    model = Sequential()
model.add(Input(shape=(img_width, img_height,3)))
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()  
keras.utils.plot_model(model, show_shapes=True) 

history = model.fit(x_train, y_train, epochs= 25, validation_data= (x_val, y_val))

# Extract the training and validation loss values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create a list of epoch numbers (1 to number of epochs)
epochs = range(1, len(train_loss) + 1)

# Plot the loss graph
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Extract the training and validation loss values from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Create a list of epoch numbers (1 to number of epochs)
epochs = range(1, len(train_accuracy) + 1)

# Plot the loss graph
plt.plot(epochs, train_accuracy , label='Training Acc')
plt.plot(epochs, val_accuracy, label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)

# Convert one-hot encoded labels back to categorical labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)

# Define labels for the confusion matrix
labels = ['Normal', 'Pneumonia']

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

