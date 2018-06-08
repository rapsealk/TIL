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
import scipy.io.wavfile as wav

from random import shuffle

class Model():

	def __init__(self):
		self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }

		self.num_sample = 32
		self.epochs = 1

		self.graph = tf.get_default_graph()

		with self.graph.as_default():

			self.input_size = 26 + 1

			self.model = Sequential()
			self.model.add(Conv2D(32, (2, 2), input_shape=(self.num_sample, self.input_size, 1), padding='same', activation='relu'))
			self.model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
			self.model.add(MaxPool2D(pool_size=(2, 2)))

			self.model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
			self.model.add(Conv2D(32, (2, 2), padding = 'same', activation='relu'))
			self.model.add(Flatten())
			self.model.add(Dense(1024, activation='relu'))
			# TODO(raise ValueError: Tensor("dense_2/Softmax:0", shape=(?, 5), dtype=float32) is not an element of this graph.)
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

			val = filter_bank_feature[:self.num_sample, :].tolist()
			for i in range(len(val)):
				val[i].append(float(0))

			processed_x.append(np.array(val))
			# processed_x.append(filter_bank_feature[:self.num_sample, :])
			processed_y.append(self.EMOTIONS[target_emotion])

			#processed_x[-1][0].append(float(0))
			#processed_x[-1][1].append(float(0))
			# print('\nemotion:', target_emotion)
			# print(filter_bank_feature[:self.num_sample, :])

		processed_x = np.array(processed_x).reshape(-1, self.num_sample, self.input_size, 1)
		processed_y = np_utils.to_categorical(processed_y)

		train_data_ratio = 0.9
		num_train_data = int(len(datalist) * train_data_ratio)

		self.train_data_x = processed_x[:num_train_data]
		self.train_data_y = processed_y[:num_train_data]

		self.test_data_x = processed_x[num_train_data:]
		self.test_data_y = processed_y[num_train_data:]


	def train(self):
		with self.graph.as_default():
			hist = self.model.fit(self.train_data_x, self.train_data_y, epochs=self.epochs, batch_size=32)
			print('>> training result')
			print(hist)

	def test(self):
		loss_and_metrics = self.model.evaluate(self.test_data_x, self.test_data_y, batch_size=32)
		print('>> test result')
		print(loss_and_metrics)

	def predict(self, x):

		with self.graph.as_default():
			x = np.array([x]).reshape(-1, self.num_sample, 26, 1)
			# x = self.train_data_x[:2]

			# self.model._make_predict_function()
			prediction = self.model.predict(x, batch_size=32)

			print('prediction:', prediction)
			print(type(prediction))

			return {
				'happy': float(prediction[0][0]),
				'neutral': float(prediction[0][1]),
				'sad': float(prediction[0][2]),
				'angry': float(prediction[0][3]),
				'disgust': float(prediction[0][4])
			}

	def save(self):
		self.model.save('cnn_emotion.h5')
		# self.model.save_weights('cnn_emotion_weights.h5')
		# arch_file = open('cnn_emotion_model.json', 'w')
		# arch_file.write(self.model.to_json())
		# arch_file.close()

	def load(self):
		self.model = load_model('cnn_emotion.h5')