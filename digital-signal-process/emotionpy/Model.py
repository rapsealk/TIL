import tensorflow as tf
import numpy as np
import os

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
from speechpy import preprocessor
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt

from random import shuffle

class Model():

	def __init__(self):
		self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3 }

		self.num_sample = 32	# 60
		self.n_mfcc = 26
		self.batch_size = 32
		self.epochs = 20

		self.graph = tf.get_default_graph()

		with self.graph.as_default():

			self.model = Sequential()
			self.model.add(Conv2D(32, (2, 2), input_shape=(self.num_sample, self.n_mfcc, 1), padding='same', activation='relu'))
			self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
			self.model.add(MaxPool2D(pool_size=(2, 2)))

			self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
			self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
			
			self.model.add(MaxPool2D(pool_size=(2, 2)))
			self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
			self.model.add(Conv2D(32, (2, 2), padding='same', activation='softmax'))
			self.model.add(MaxPool2D(pool_size=(4, 4)))
			self.model.add(Flatten())
			#self.model.add(Dropout(rate=0.75))

			self.model.add(Dense(128, activation='relu'))
			self.model.add(Dense(len(self.EMOTIONS), activation='sigmoid'))

			# 모델 학습과정
			self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	def load_dataset(self, csv='./dataset/firebase.csv'):
	# def load_dataset(self, csv='./dataset/tess/dataset.csv'):
		handle = open(csv, 'r')
		rawlines = handle.read().split('\n')
		handle.close()
		datalist = [line.split(',') for line in rawlines]
		shuffle(datalist)

		processed_x = []
		processed_y = []

		for url, target_emotion, alpha in datalist:
			# if filename in ['YAF_germ_angry.wav']: continue

			# rate, signal = wav.read('./dataset/tess/' + target_emotion + '/' + filename)
			# filter_bank_feature = log_filter_bank(signal, rate)
			filter_bank_feature = preprocessor.load_and_process(url)

			processed_x.append(filter_bank_feature)
			# processed_x.append(filter_bank_feature[:self.num_sample, :])
			processed_y.append(self.EMOTIONS[target_emotion])

		processed_x = np.array(processed_x).reshape(-1, self.num_sample, self.n_mfcc, 1)
		processed_y = np_utils.to_categorical(processed_y)

		train_data_ratio = 0.9
		num_train_data = int(len(datalist) * train_data_ratio)

		self.train_data_x = processed_x[:num_train_data]
		self.train_data_y = processed_y[:num_train_data]

		self.test_data_x = processed_x[num_train_data:]
		self.test_data_y = processed_y[num_train_data:]


	def train(self):

		self.load_dataset()

		with self.graph.as_default():
			hist = self.model.fit(self.train_data_x, self.train_data_y, epochs=self.epochs, batch_size=self.batch_size)
			print('>> training result')
			print(hist)
			self.histogram(hist)

	def test(self):
		loss_and_metrics = self.model.evaluate(self.test_data_x, self.test_data_y, batch_size=self.batch_size)
		print('>> test result')
		print(loss_and_metrics)

	def predict(self, x):

		with self.graph.as_default():
			x = np.array([x]).reshape(-1, self.num_sample, self.n_mfcc, 1)
			# x = self.train_data_x[:2]

			# self.model._make_predict_function()
			prediction = self.model.predict(x, batch_size=self.batch_size)

			print('prediction:', prediction)
			print(type(prediction))

			return {
				'happy': float(prediction[0][0]),
				'neutral': float(prediction[0][1]),
				'sad': float(prediction[0][2]),
				'angry': float(prediction[0][3]),
				#'disgust': float(prediction[0][4])
			}

	def predict_with_test_set(self, n=100):
		with self.graph.as_default():
			x = np.array([self.test_data_x[:n]]).reshape(-1, self.num_sample, self.n_mfcc, 1)
			pred = self.model.predict(x, batch_size=self.batch_size)

			argx = {}
			argx[True] = argx[False] = 0

			for i, pred in enumerate(prediction):
				print('[{}] label: {}'.format(i+1, self.test_data_y[i]))
				print('result:', {
					'happy': float(pred[0]),
					'neutral': float(pred[1]),
					'sad': float(pred[2]),
					'angry': float(pred[3])
				})
				argx[np.argmax(self.test_data_y[i]) == np.argmax(pred)] += 1
			print(argx)

	def save(self):
		self.model.save('cnn_emotion.h5')

	def load(self):
		self.model = load_model('cnn_emotion.h5')

	def histogram(self, hist):
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


if __name__ == '__main__':
	model = Model()
	model.load_dataset()
	model.train()