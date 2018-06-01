import tensorflow as tf
import numpy as np
import os

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from random import shuffle

class Model():

	def __init__(self):
		self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }

		self.num_sample = 32
		
		"""
		self.X = tf.placeholder(tf.float32, [None, self.num_sample, 26, 1])
		self.Y = tf.placeholder(tf.float32, [None, len(self.EMOTIONS)])
		self.keep_prob = tf.placeholder(tf.float32)

		self.W1 = tf.Variable(tf.random_normal([2, 2, 1, 32], stddev=0.01))
		self.L1 = tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding='SAME')
		self.L1 = tf.nn.relu(self.L1)

		self.W2 = tf.Variable(tf.random_normal([2, 26, 32, 32], stddev=0.01))
		# self.W2 = tf.Variable(tf.random_normal([2, 2, 1, 32], stddev=0.01))
		# TODO(resize)
		self.L2 = tf.nn.conv2d(self.L1, self.W2, strides=[1, 1, 1, 1], padding='SAME')
		self.L2 = tf.nn.relu(self.L2)
		self.L2 = tf.nn.max_pool(self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		self.W3 = tf.Variable(tf.random_normal([2, 13, 32, 32], stddev=0.001))
		# self.W3 = tf.Variable(tf.random_normal([2, 2, 1, 32], stddev=0.01))
		self.L3 = tf.nn.conv2d(self.L2, self.W3, strides=[1, 1, 1, 1], padding='SAME')
		self.L3 = tf.nn.relu(self.L3)

		self.W4 = tf.Variable(tf.random_normal([2, 13, 32, 32], stddev=0.01))
		self.L4 = tf.nn.conv2d(self.L3, self.W4, strides=[1, 1, 1, 1], padding='SAME')
		self.L4 = tf.nn.relu(self.L4)

		self.F1 = tf.layers.flatten(self.L4)
		self.D1 = tf.layers.dense(self.F1, 1024, activation=tf.nn.relu)
		self.model = tf.layers.dense(self.D1, len(self.EMOTIONS), activation=tf.nn.softmax)

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.model, labels=self.Y))
		self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

		self.init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(self.init)

		self.batch_size = 32

		self.epoch = 3 # 30
		"""
		
	
	def model_function(self, features, targets, mode=None, params=None):
		# self.num_sample = 32

		self.num_of_outputs = 10
		self.L1 = tf.contrib.layers.relu(features, self.num_of_outputs)
		self.L2 = tf.contrib.layers.reul(self.L1, self.num_of_outputs)
		self.L3 = tf.contrib.layers.linear(self.L2, len(self.EMOTIONS))

		predictions = tf.reshape(self.L3, [-1])
		predictions_dict = { key: [] for key in self.EMOTIONS }
		for key, val in self.EMOTIONS:
			predictions_dict[key].append(predictions[val])

		loss = tf.contrib.losses.mean_squared_error(predictions, targets)

		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=params['learning_rate'],
			optimizer='SGD'
		)

		return predictions_dict, loss, train_op


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

			_x = filter_bank_feature[:self.num_sample, :]
			_x = np.reshape([_x], [-1, self.num_sample, 26, 1])
			processed_x.append(_x)
			# processed_x.append(filter_bank_feature[:self.num_sample, :])
			# print('x.shape:', np.shape(_x))
			# processed_y.append(self.EMOTIONS[target_emotion])
			_y = [0, 0, 0, 0, 0]
			_y[self.EMOTIONS[target_emotion]] = 1
			_y = np.reshape([_y], [-1, 5])
			processed_y.append(_y)
			# print('\nemotion:', target_emotion)
			# print(filter_bank_feature[:self.num_sample, :])

		# processed_x = np.array(processed_x).reshape(-1, self.num_sample, 26, 1)
		# processed_y = np_utils.to_categorical(processed_y)

		self.datalist_size = len(datalist)

		train_data_ratio = 0.9
		self.num_train_data = int(self.datalist_size * train_data_ratio)
		self.num_test_data = self.datalist_size - self.num_train_data

		self.train_data_x = processed_x[:self.num_train_data]
		self.train_data_y = processed_y[:self.num_train_data]

		self.test_data_x = processed_x[self.num_train_data:]
		self.test_data_y = processed_y[self.num_train_data:]


	def train(self):
		for epoch in range(self.epoch):
			total_cost = 0
			for i in range(self.num_train_data):
				batch_x = self.train_data_x[i]
				batch_y = self.train_data_y[i]
				# reshape
				_, cost_val = self.sess.run([self.optimizer, self.cost], feed_dict={ self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.7 })
				total_cost += cost_val

			print('Epoch:', '%04d / %04d' % (epoch + 1, self.epoch), 'Avg. cost = ', '{:.3f}'.format(total_cost / self.datalist_size))

	
	def test(self):
		is_correct = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
		accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
		print('Accuracy:', self.sess.run(accuracy, feed_dict={ self.X: self.test_data_x, self.Y: self.test_data_y, self.keep_prob: 1 }))

	"""
	def predict(self, x):
		features = { X: np.array(x) }

		feature_columns = []
		for key, val in self.EMOTIONS:
			feature_columns.append(key)
		self.classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns)
		template = 'Prediction is "{}" ({:.1f}%)'
	"""

