#!/usr/bin/env python3
"""
YOLO: You Only Look Once
arxiv paper: https://arxiv.org/pdf/1506.02640.pdf
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np

from keras import layers
from keras import models
from keras.applications import vgg16
#from keras.applications import inception_v3, resnet50, mobilenet
from keras import losses

from keras.layers import advanced_activations

class YOLODetectionNetwork(object):

    def __init__(self):
        self.model = models.Sequential()

        # From Block #1 to Block #4 - ImageNet

        self.model.add(vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))

        #""" Block #1 """
        #self.model.add(layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), activation=advanced_activations.LeakyReLU, input_size=(224, 224, 3)))
        #self.model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        #""" Block #2 """
        #self.model.add(layers.Conv2D(filters=192, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        #""" Block #3 """
        #self.model.add(layers.Conv2D(filters=128, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        #""" Block #4 """
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=256, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU))
        #self.model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        """ Block #5
        self.model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), input_shape=(None, 7, 7, 512), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(2, 2), strides=(2, 2), activation=advanced_activations.LeakyReLU()))
        """

        """ Block #5 """
        self.model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), input_shape=(None, 7, 7, 512), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=512, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))

        """ Block #6
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=advanced_activations.LeakyReLU()))
        # Linear activation function for the final layer
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation='relu'))
        """

        """ Block #6 """
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation=advanced_activations.LeakyReLU()))
        # Linear activation function for the final layer
        self.model.add(layers.Conv2D(filters=1024, kernel_size=(1, 1), activation='relu'))

        """ Block #7 """
        self.model.add(layers.Dense(units=1024))
        #self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=30))

        self.model.summary()

        """ Compile """
        self.model.compile(optimizer='adam', loss=losses.mean_squared_error, metrics=['acc'])


if __name__ == '__main__':
    detector = YOLODetectionNetwork()