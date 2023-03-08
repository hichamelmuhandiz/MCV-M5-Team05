import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import tensorflow_model_optimization as tfmot

# Residual units from the ResNet paper
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Fire units from the SqueezeNet paper
class FireUnit(keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, s_1, e_1, e_2, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.squeeze = keras.layers.Conv2D(s_1, 1, strides=1, padding='same')
        self.expand_1 = keras.layers.Conv2D(e_1, 1, strides=1, padding='same')
        self.expand_2 = keras.layers.Conv2D(e_2, 3, strides=1, padding='same')
    
    def get_prunable_weights(self):
        # Prune bias and kernel of layer
        return [self.squeeze.kernel, self.expand_1.kernel, self.expand_2.kernel, self.squeeze.bias, self.expand_1.bias, self.expand_2.bias]
    
    def call(self, inputs):
        Z = inputs
        Z = self.squeeze(Z)
        Z = self.activation(Z)
        Z_e1 = self.expand_1(Z)
        Z_e1 = self.activation(Z_e1)
        Z_e2 = self.expand_2(Z)
        Z_e2 = self.activation(Z_e2)
        return keras.layers.Concatenate()([Z_e1, Z_e2])


# MLP classifier
def create_mlp_classifier():
    model = keras.models.Sequential()
    model.add(keras.layers.Reshape((224*224*3,),input_shape=(224, 224, 3),name='first'))
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model

# Convolutional neural network
def create_conv_classifier():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model

# Small resnet with spatial dropout
def create_resnet_model_small_spatial_dropout():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 3, strides=2, input_shape=[224, 224, 3], padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.SpatialDropout2D(0.5))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    for filters in [32] * 1 + [16] * 1:
        strides = 1
        model.add(ResidualUnit(filters, strides=strides))
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Dense(8, activation='softmax', activity_regularizer=keras.regularizers.L1()))
    return model

# Small resnet
def create_resnet_model_small():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    prev_filters = 64
    for filters in [64] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model

# ResNet-34
def create_resnet_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding='same', use_bias=False, activity_regularizer=keras.regularizers.L1()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        model.add(keras.layers.SpatialDropout2D(0.2))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(8, activation='softmax', kernel_regularizer=keras.regularizers.L1(0.01),
    #                  activity_regularizer=keras.regularizers.L2(0.01)))
    # model.add(keras.layers.Dropout(.5))
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model

# SqueezeNet model 
def create_squeeze_net_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(96, 7, strides=2, input_shape=[224, 224, 3], padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    
    params_s_1 = [16, 16, 32]
    params_e_1 = [64, 64, 128]
    params_e_2 = [64, 64, 128]
    for i in range(len(params_s_1)):
        model.add(FireUnit(params_s_1[i], params_e_1[i], params_e_2[i]))
    
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    
    params_s_1 = [32, 48, 48, 64]
    params_e_1 = [128, 192, 192, 256]
    params_e_2 = [128, 192, 192, 256]
    for i in range(len(params_s_1)):
        model.add(FireUnit(params_s_1[i], params_e_1[i], params_e_2[i]))
        
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(FireUnit(64, 256, 256))
    model.add(keras.layers.Conv2D(1000, 1, strides=1, padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(8, activation='softmax'))
    return model