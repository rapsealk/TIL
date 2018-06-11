import tensorflow as tf
import numpy as np
import os

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D
from keras.layers import Flatten, Dropout

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from speechpy import preprocessor
from speechpy.raw_socket import TCPSocket

from random import shuffle
from time import time

class Model():

	def __init__(self):
		self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3 }

		self.num_sample = 32
		self.epochs = 30

		# self.event_queue = queue.Queue()

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

			# train with 0 value
			val = filter_bank_feature[:self.num_sample, :].tolist()
			for i in range(len(val)):
				val[i].append(float(0))

			print('local:', np.shape(np.array(val)))

			processed_x.append(np.array(val))
			processed_y.append(self.EMOTIONS[target_emotion])

		processed_x = np.array(processed_x).reshape(-1, self.num_sample, self.input_size, 1)
		processed_y = np_utils.to_categorical(processed_y)

		train_data_ratio = 0.9
		num_train_data = int(len(datalist) * train_data_ratio)

		self.train_data_x = processed_x[:num_train_data]
		self.train_data_y = processed_y[:num_train_data]

		self.test_data_x = processed_x[num_train_data:]
		self.test_data_y = processed_y[num_train_data:]


	def load_dataset_for_gapi(self, external=True, csv='/dataset/firebase.csv'):

		handle = open(os.path.dirname(__file__) + csv, 'r')
		datalist = [line.replace('\n', '').split(',') for line in handle.readlines()]
		handle.close()

		shuffle(datalist)

		processed_x = []
		processed_y = []

		#recorder = open(os.path.dirname(__file__) + '/dataset/i' + str(time()).split('.')[0] + '.csv', 'w')

		for url, target_emotion, alpha in datalist:

			if external:
				sockett = TCPSocket()
				ADDRESS = '192.168.195.186'
				sockett.connect(ADDRESS, 5000)
				print('url:', url)
				sockett.send(url + '\n')
				alpha = float(sockett.receive())
				sockett.close()
				# sockett = TCPSocket()
				# alpha = sockett.read_gapi(url)
				filter_bank_feature = preprocessor.download_and_process(url)
			else:
				alpha = float(alpha)
				filter_bank_feature = preprocessor.load_and_process(url)

			#recorder.write('{},{},{}\n'.format(url, target_emotion, alpha))

			val = filter_bank_feature[:self.num_sample, :].tolist()
			for i in range(len(val)):
				val[i].append(alpha)

			print('gapii:', np.shape(np.array(val)), url)

			processed_x.append(np.array(val))
			processed_y.append(self.EMOTIONS[target_emotion])

		#recorder.close()

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
			self.hist = self.model.fit(self.train_data_x, self.train_data_y, epochs=self.epochs, batch_size=32)
			print('>> training result')
			print(self.hist)
			print('history.loss:', self.hist.history['loss'])
			print('history.acc:', self.hist.history['acc'])
		
			try:
				print('>> a')
				fig, loss_ax = plt.subplots()
				print('>> b')
				acc_ax = loss_ax.twinx()
				print('>> c')
				loss_ax.plot(self.hist.history['loss'], 'y', label='train loss')
				print('>> d')
				acc_ax.plot(self.hist.history['acc'], 'b', label='train acc')
				print('>> e')
				loss_ax.set_xlabel('epoch')
				print('>> f')
				loss_ax.set_ylabel('loss')
				print('>> g')
				acc_ax.set_ylabel('accuray')
				print('>> h')
				loss_ax.legend(loc='upper left')
				print('>> i')
				acc_ax.legend(loc='lower left')
				print('>> plot')
				
				plt.show()
				# fig.savefig('{}.png'.fotmat(time()))

			except: # Error as e:
				print('>> Exception')
				# print(e)


	def test(self):
		with self.graph.as_default():
			loss_and_metrics = self.model.evaluate(self.test_data_x, self.test_data_y, batch_size=32)
			print('>> test result')
			print(loss_and_metrics)

	def predict(self, x, alpha):
		with self.graph.as_default():
			x = x.tolist()
			for i in range(len(x)):
				x[i].append(alpha)

			x = np.array([x]).reshape(-1, self.num_sample, self.input_size, 1)

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

	def load(self):
		self.model = load_model('cnn_emotion.h5')