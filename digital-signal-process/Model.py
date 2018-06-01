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

class Model():

	def __init__(self):
		self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }

		self.num_sample = 32

		self.model = Sequential()
		self.model.add(Conv2D(32, (2, 2), input_shape=(self.num_sample, 26, 1), padding='same', activation='relu'))
		self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
		self.model.add(MaxPool2D(pool_size=(2, 2)))

		self.model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
		self.model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
		self.model.add(Flatten())
		self.model.add(Dense(1024, activation='relu'))
		self.model.add(Dense(len(self.EMOTIONS), activation='softmax'))

		# 모델 학습과정
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	def load_dataset(self, csv='./dataset/tess/dataset.csv'):
		handle = open(csv, 'r')
		rawlines = handle.read().split('\n')
		handle.close()
		datalist = [line.split(',') for line in rawlines]
		shuffle(datalist)

		processed_x = []
		processed_y = []

		for filename, target_emotion in datalist:
			if filename in ['YAF_germ_angry.wav']: continue

			rate, signal = wav.read('./dataset/tess/' + target_emotion + '/' + filename)
			filter_bank_feature = log_filter_bank(signal, rate)

			processed_x.append(filter_bank_feature[:self.num_sample, :])
			processed_y.append(self.EMOTIONS[target_emotion])
			print('\nemotion:', target_emotion)
			print(filter_bank_feature[:self.num_sample, :])

		processed_x = np.array(processed_x).reshape(-1, self.num_sample, 26, 1)
		processed_y = np_utils.to_categorical(processed_y)

		train_data_ratio = 0.9
		num_train_data = int(len(datalist) * train_data_ratio)

		self.train_data_x = processed_x[:num_train_data]
		self.train_data_y = processed_y[:num_train_data]

		self.test_data_x = processed_x[num_train_data:]
		self.test_data_y = processed_y[num_train_data:]


	def train(self):
		hist = self.model.fit(self.train_data_x, self.train_data_y, epochs=30, batch_size=32)
		print('>> training result')
		print(hist)

	def test(self):
		loss_and_metrics = self.model.evaluate(self.test_data_x, self.test_data_y, batch_size=32)
		print('>> test result')
		print(loss_and_metrics)

	def save(self):
		self.model.save('cnn_emotion_model.h5')

""" 그래프 표시
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
"""

if __name__ == '__main__':
	model = Model()
	model.load_dataset()
	model.train()
	model.test()
	model.save()