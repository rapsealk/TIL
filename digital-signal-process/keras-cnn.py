import tensorflow as tf
import numpy as np
import os

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout

import matplotlib.pyplot as plt

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from random import shuffle

EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }
DATA_STATIC_PATH = './dataset/tess/'

# EMOTIONS = { 'happy': 0, 'neutral': 1, 'nervous': 2, 'sad': 3, 'angry': 4 }
# DATA_STATIC_PATH

""" Inside Out
EMOTIONS = ['happy', 'sad', 'nervous', 'angry']
DATA_STATIC_PATH = './dataset/inside_out/' # './dataset/tess/'
"""

"""
학습 데이터를 csv 파일에서 읽어온 다음,
[(파일 이름, 감정)] 형태로 반환.
"""
def load_dataset(csv_file='./dataset/tess/dataset.csv'):
    handle = open(csv_file, 'r')
    rawlines = handle.read().split('\n')#[:-1]
    handle.close()
    datalist = [line.split(',') for line in rawlines]
    return datalist

#데이터 셔플
datalist = load_dataset()
shuffle(datalist)

train_data_ratio = 0.9
num_train_data = int(len(datalist) * train_data_ratio)
train_datalist = datalist[:num_train_data]
test_datalist = datalist[num_train_data:]

x_train = []
y_train = []
num_sample = 32

for filename, target_emotion in train_datalist:
    if filename in ['YAF_germ_angry.wav']: continue
    #mfcc 처리
    rate, signal = wav.read(DATA_STATIC_PATH + target_emotion + '/' + filename)
    filter_bank_feature = log_filter_bank(signal, rate)

    # 트레이닝 데이터 생성
    x_train.append(filter_bank_feature[:num_sample, :])
    # x_train.append(filter_bank_feature[1:3, :]) 
    y_train.append(EMOTIONS[target_emotion])

x_train = np.array(x_train).reshape(-1, num_sample, 26, 1)
y_train = np_utils.to_categorical(y_train)

x_test = []
y_test = []

for filename, target_emotion in test_datalist:
    if filename in ['YAF_germ_angry.wav']: continue
    
    # mfcc 처리
    rate, signal = wav.read(DATA_STATIC_PATH + target_emotion + '/' + filename)
    #mfcc_feature = mfcc(signal, rate)
    #d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)

    # 테스트 데이터 생성
    x_test.append(filter_bank_feature[:num_sample, :])
    # x_test.append(filter_bank_feature[1:3, :]) 
    y_test.append(EMOTIONS[target_emotion])

x_test = np.array(x_test).reshape(-1, num_sample, 26, 1)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(num_sample, 26, 1), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

"""
model.add(Conv2D(32, (2, 2), input_shape=(1, 13, 32), padding = 'same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
model.add(MaxPool2D(pool_size=(1, 2)))
"""
model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(len(EMOTIONS), activation='softmax'))

#모델 학습과정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#모델 트레이닝
hist = model.fit(x_train, y_train, epochs=30, batch_size=32)
print('## training result ##')
print(hist)

#모델 평가
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

#모델 저장
model.save('cnn_emotion_model.h5')

#그래프 표시
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

#
#test_fig, test_loss_ax = plt.subplots()
#test_acc_ax = test_loss_ax.twinx()
#test_loss_ax.plot()