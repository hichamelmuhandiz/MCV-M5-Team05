import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models_keras import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

import wandb
from wandb.keras import WandbMetricsLogger

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

### HYPERPARAMETERS
train_set = 'MIT_SPLIT'
root_dir = './datasets/' + train_set

train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height=224
batch_size=4
epochs = 1000

#initialise wandb
log_wandb = False    #whether or not to log with wandb
if log_wandb: writer = wandb.init(name="keras_smallMIT_SPLIT_batch4", project="week-1", config={"lr":0.1})
### CREATE DATASET
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
  vertical_flip=True,
  rotation_range=90,)

itr = train_datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=4,
class_mode='categorical')
itr.reset()
X_train, y_train = itr.next()
print(X_train.shape, y_train.shape)

test_datagen = ImageDataGenerator()
itr = test_datagen.flow_from_directory(
test_data_dir,
target_size=(img_width, img_height),
batch_size=4,
class_mode='categorical')
itr.reset()
X_test, y_test = itr.next()
print(X_test.shape, y_test.shape)

train_datagen_tmp = ImageDataGenerator(
  horizontal_flip=True,
  vertical_flip=True,
  rotation_range=90,
)

test_datagen_tmp = ImageDataGenerator(
)



train_generator = train_datagen_tmp.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])

validation_generator = test_datagen_tmp.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['coast', 'forest', 'highway', 'inside_city',
                 'mountain', 'Opencountry', 'street', 'tallbuilding'])





# OUR SMALL SQUEEZENET
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, 3, strides=2, input_shape=[224, 224, 3]))
model.add(FireUnit(8, 16, 16))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Dense(8, activation='softmax'))



# TRAINING WITH DIFFERENT METRICS
print(model.summary())
print(model.count_params())
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0)
opt = tf.keras.optimizers.Adadelta(learning_rate=0.1)
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

if log_wandb:
    history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, validation_data=validation_generator,callbacks=[WandbMetricsLogger()])
else:
    history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, validation_data=validation_generator)

# SAVING RESULTS
offset = 100
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),history.history['accuracy'][::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),history.history['val_accuracy'][::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Accuracy',fontsize=25)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./Week 1/accuracy_keras.jpg',transparent=False)
plt.close()

plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),history.history['loss'][::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),history.history['val_loss'][::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./Week 1/loss_keras.jpg',transparent=False)
plt.close()