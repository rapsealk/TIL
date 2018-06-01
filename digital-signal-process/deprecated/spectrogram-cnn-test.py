#/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from scipy.misc import imread
import time

from random import shuffle

EMOTIONS = ['happy', 'neutral', 'sad', 'angry', 'disgust']
DATA_STATIC_PATH = './dataset/tess/spectrogram/'

""" Inside Out
EMOTIONS = ['happy', 'sad', 'nervous', 'angry']
DATA_STATIC_PATH = './dataset/inside_out/'
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

def onehot_output(target_emotion):
    output = [0 for i in range(len(EMOTIONS))]
    output[EMOTIONS.index(target_emotion)] = 1
    return output

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

DEPTH = 4

x = tf.placeholder(tf.float32, shape=[480, 640, DEPTH], name='x')
# x = tf.placeholder(tf.float32, shape=[129, 129])
y = tf.placeholder(tf.float32, shape=[None, len(EMOTIONS)], name='y')

x_spectrogram = tf.reshape(x, [-1, 480, 640, DEPTH], name='x_spectrogram')
# x_spectrogram = tf.reshape(x, [-1, 129, 129, 1])

W_conv1 = tf.Variable(tf.truncated_normal([2, 2, DEPTH, 32], stddev=0.1), name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')
# W_conv1 = weight_variable([2, 2, DEPTH, 32])
# b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_spectrogram, W_conv1, strides=[1, 2, 2, 1], padding='SAME', name='conv2d_1') + b_conv1, name='h_conv1')
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
# h_conv1 = tf.nn.relu(conv2d(x_spectrogram, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1), name='W_conv2')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[2, 2, 32, 64]), name='b_conv2')
# W_conv2 = weight_variable([2, 2, 32, 64])
# b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(x_spectrogram, W_conv2, strides=[1, 2, 2, 1], padding='SAME', name='conv2d_2') + b_conv2, name='h_conv2')
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = tf.Variable(tf.truncated_normal([30 * 40 * 64, 1024], stddev=0.1), name='W_fc1')
b_fc1 = tf.Variable(tf.constant(0.1, [1024]), name='b_fc1')
# W_fc1 = weight_variable([30 * 40 * 64, 1024])
# b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 30 * 40 * 64], name='h_pool2_flat')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name='h_fc1')
# h_pool2_flat = tf.reshape(h_pool2, [-1, 30 * 40 * 64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = tf.Variable(tf.truncated_normal([1024, len(EMOTIONS)], stddev=0.1), name='W_fc2')
b_fc2 = tf.Variable(tf.constant(0.1, [len(EMOTIONS)]), name='b_fc2')
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')
# W_fc2 = weight_variable([1024, len(EMOTIONS)])
# b_fc2 = bias_variable([len(EMOTIONS)])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train and evaluation
cross_entropy = -1 * tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1), name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# cross_entropy = -1 * tf.reduce_sum(y * tf.log(y_conv))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './history/spectrogram-cnn.ckpt')
print('Spectrogram CNN restored.')

datalist = load_dataset()
shuffle(datalist)

timestamp = time.time()
test_epoch = 0
test_acc = []

for filename, target_emotion in datalist:
    test_epoch += 1
    if filename in ['YAF_germ_angry.wav']: continue
    
    filepath = DATA_STATIC_PATH + filename.split('.')[0] + '.png'
    image = imread(filepath)

    emotion_vector = [onehot_output(target_emotion)]
    # print('test filename:', filename)
    # print('expected:', emotion_vector, 'actual:', target_emotion, 'rate:', rate)

    _timestamp = time.time()
    test_accuracy = sess.run(accuracy, feed_dict={ x: image, y: emotion_vector, keep_prob: 1.0 })
    print('epoch %d' %test_epoch, test_accuracy, '- %fs' %(_timestamp - timestamp), filename)
    test_acc.append(test_accuracy)
    timestamp = _timestamp

print('total test_accuracy: %g' %(sum(test_acc) / len(datalist)))