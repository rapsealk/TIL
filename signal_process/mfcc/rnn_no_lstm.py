import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

from mfcc import mfcc
from mfcc import delta
from mfcc import log_filter_bank
import scipy.io.wavfile as wav

import os

############################################################################################
# https://github.com/golbin/TensorFlow-ML-Exercises/blob/master/12%20-%20RNN/00_no_lstm.py #
############################################################################################

"""
root, directory, files = os.walk('.')
print('root:', root)
print('directory:', directory)
print('files:', files)

directories = root[1]
directories.remove('__pycache__')
directories.remove('.vscode')
print('directories:', directories)

exit(0)
"""

handle = open('./dataset.csv', 'r')
raw_lines = handle.read().split('\n')
handle.close()
lines = [(lambda x: x.split(','))(line) for line in raw_lines]
print(lines)

batch_size = 1

for filename, target_emotion in lines:
    rate, signal = wav.read(filename)
    mfcc_feature = mfcc(signal, rate)
    d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)
    print(filter_bank_feature[1:3, :])
    sequence_length = len(filter_bank_feature[1]) - 1
    print('sequence_length:', sequence_length)

sample = " if you want you"
sample_idx = [2, 4, 6, 7, 9]
print('sample_idx:', sample_idx)
print('sample_idx[:-1]:', sample_idx[:-1])
print('sample_idx[1:]:', sample_idx[1:])

exit(0)

#################################
# Golbin Tensorflow RNN-NO-LSTM #
#################################

# sample = " if you want you"

# char_set = list(set(sample))  # id -> char
# char_dic = { w: i for i, w in enumerate(char_set) }

# - settings -
# rnn_hidden_size = dic_size = len(char_dic)  # output size of each cell
# batch_size = 1  # one sample data,one batch
# sequence_length = len(sample) - 1  # number of lstm rollings (unit #)

# sample_idx = [char_dic[c] for c in sample]  # char to index
x_data = tf.one_hot(sample_idx[:-1], dic_size)  # one hot
y_data = sample_idx[1:]

# split to input (char)length. This will decide unrolling size
x_split = tf.split(value=x_data, num_or_size_splits=[sequence_length])
outputs = x_split # No lstm

# softmax layer
softmax_w = tf.get_variable("softmax_w", [sequence_length, dic_size])
softmax_b = tf.get_variable("softmax_b", [dic_size])
outputs = outputs * softmax_w + softmax_b

outputs = tf.reshape(outputs, [batch_size, sequence_length, dic_size])
y_data = tf.reshape(y_data, [batch_size, sequence_length])
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(outputs, y_data, weights)
mean_loss = tf.reduce_mean(sequence_loss) / batch_size
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run([train_op, mean_loss])
    results = sess.run(outputs)
    for result in results:
        index = np.argmax(result, axis=1)
        print(''.join([char_set[t] for t in index]), l)