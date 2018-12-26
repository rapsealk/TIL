#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import h5py
import numpy as np

batch_point = 0

def next_batch(dataset, batch_size):
    global batch_point
    if batch_point + batch_size < len(dataset['input']):
        x = dataset['input'][batch_point:batch_point+batch_size]
        y_ = dataset['output'][batch_point:batch_point+batch_size]
        batch_point += batch_size
    else:
        x = dataset['input'][batch_point:]
        y_ = dataset['output'][batch_point:]
        batch_point = 0
    y = onehot(y_)
    return x, y

def onehot(label):
    y = []
    for i in range(len(label)):
        temp = np.zeros(10, dtype=int)
        temp[label[i]] = 1
        y.append(list(temp))
    y = np.array(y)
    y = np.reshape(y, (len(y), 10))
    return y

dataset = h5py.File('mnist_raw.hdf5', 'r')

# Build a model.
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

saver = tf.train.Saver({'W': W, 'b': b})
sess = tf.Session()
saver.restore(sess, './variables.ckpt')

for _ in range(10):
    batch_xs, batch_ys = next_batch(dataset['train'], 100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

saver = tf.train.Saver({'W': W, 'b': b})
saver.save(sess, './variables.ckpt')
#print(sess.run(b))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print('Centralized accuracy:', sess.run(accuracy, feed_dict={x: dataset['validation']['input'], y_: onehot(dataset['validation']['output'])}))