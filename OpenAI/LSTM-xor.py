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

def RandomBinaryString(length=50):
    from random import randint
    return ''.join([str(randint(0, 1)) for i in range(50)])

def ComputeParityBit(bits):
    pbit = int(bits[0])
    for i in range(len(bits)-1):
        pbit ^= int(bits[i+1])
    return pbit

class LSTMXOR():

    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(units=32, activiaon='relu'))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, dataset):
        labels = [[ComputeParityBit(data)] for data in dataset]
        hist = self.model.fit(dataset, labels, epochs=10, batch_size=32)
        print(hist['hist'])


if __name__ == '__main__':
    model = LSTMXOR()
    dataset = [RandomBinaryString() for i in range(100000)]
    model.train(dataset)