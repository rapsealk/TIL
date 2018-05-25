#/usr/bin/python3
# -*- coding: utf-8 *-

########################################################
# https://tykimos.github.io/2017/04/09/RNN_Layer_Talk/ #
########################################################

import tensorflow as tf
import numpy as np
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import np_utils

from speechpy import mfcc
#from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from random import shuffle

EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }
DATA_STATIC_PATH = './dataset/tess/'

NUM_SAMPLE = 32

def load_datalist(csv_file=DATA_STATIC_PATH+'dataset.csv'):
    handle = open(csv_file, 'r')
    rawlines = handle.read().split('\n')
    handle.close()
    datalist = [line.split(',') for line in rawlines]
    # 데이터 셔플
    shuffle(datalist)

    train_data_ratio = 0.9
    num_train_data = int(len(datalist) * train_data_ratio)
    train_datalist = datalist[:num_train_data]
    test_datalist = datalist[num_train_data:]

    return train_datalist, test_datalist

def load_dataset(datalist=[]):
    x_dataset = []
    y_dataset = []
    for filename, target_emotion in datalist:
        if filename in ['YAF_germ_angry.wav']: continue
        rate, signal = wav.read(DATA_STATIC_PATH + target_emotion + '/' + filename)
        filter_bank_feature = log_filter_bank(signal, rate)
        x_dataset.append(filter_bank_feature[:NUM_SAMPLE, :])
        y_dataset.append(EMOTIONS[target_emotion])
    return x_dataset, y_dataset


if __name__ == '__main__':
    train_datalist, test_datalist = load_datalist()
    train_x, train_y = load_dataset(train_datalist)

    TIME_STEP = 4   # equals to window size
    NUM_ATTRIBUTES = 1
    # IS_STATEFUL = False

    # Build model
    model = Sequential()
    #model.add(Dense(128, input_dim=4, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(len(EMOTIONS), activation='softmax'))
    model.add(LSTM(128, input_shape=(TIME_STEP, NUM_ATTRIBUTES)))
    model.add(Dense(len(EMOTIONS), activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    loss_history = LossHistory()
    loss_history.init()