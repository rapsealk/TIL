#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
    OpenAI Research 2.0: https://blog.openai.com/requests-for-research-2/ (XOR Even Parity)
    Understanding LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    Parity bit: https://ko.wikipedia.org/wiki/%ED%8C%A8%EB%A6%AC%ED%8B%B0_%EB%B9%84%ED%8A%B8
"""

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import Sequential
from keras.layers import LSTM, Dense

import numpy as np

from random import randint

def RandomBinaryString(length=50):
    return ''.join([str(randint(0, 1)) for i in range(50)])

def ComputeParityBit(bits):
    #print('ComputeParityBit:', bits)
    pbit = int(bits[0])
    for i in range(len(bits)-1):
        pbit ^= int(bits[i+1])
    #print('pbit:', pbit)
    return pbit

class LSTMXOR():

    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(units=1024, activation='tanh'))
        #self.model.add(Dense(units=128, activation='relu'))
        #self.model.add(LSTM(units=32, activation='relu'))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, dataset):
        #labels = [[ComputeParityBit(data)] for data in dataset]
        labels = [ComputeParityBit(data) for data in dataset]
        """
        for data, label in zip(dataset, labels):
            data = [e for e in data]
            data = np.array([data]).reshape(-1, 50, 1)
            label = np.array([label]).reshape(-1, 1)
            hist = self.model.fit(data, label, epochs=1, batch_size=32)
        """
        for i, data in enumerate(dataset):
            dataset[i] = np.array([e for e in data]).reshape(-1, 50)

        dataset = np.array(dataset)
        labels = np.array(labels)

        self.model.fit(dataset, labels, epochs=10, batch_size=32)

    def validate(self, dataset):
        pbits = [ComputeParityBit(data) for data in dataset]
        dataset = [[e for e in data] for data in dataset]
        dataset = [np.array([data]).reshape(-1, 1, 50) for data in dataset]
        correct = 0
        for data, pbit in zip(dataset, pbits):
            pred = self.model.predict_classes(data)
            correct += pred[0][0] == pbit
        print('correct:', correct, '/', len(dataset), '(%f)' % (correct / len(dataset)))

if __name__ == '__main__':
    model = LSTMXOR()
    dataset = [RandomBinaryString() for i in range(100000)]
    train_dataset = dataset[:90000]
    valid_dataset = dataset[90000:]
    model.train(train_dataset)
    model.validate(valid_dataset)