#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    '--federated',
    type=bool,
    default=False,
    help='If True, MNIST data will be splitted into 3 groups for federated learning.'
)
FLAGS, unparsed = parser.parse_known_args()

W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))

init = tf.global_variables_initializer()
saver = tf.train.Saver({'W': W, 'b': b})

with tf.Session() as sess:
    sess.run(init)
    checkpoint = './mean_variables.ckpt' if FLAGS.federated else './variables.ckpt'
    saver.save(sess, checkpoint)