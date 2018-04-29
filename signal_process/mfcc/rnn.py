#!/usr/bin/python3
# -*- coding: utf-8 -*-

#############################################################
# Reference:  https://www.svds.com/tensorflow-rnn-tutorial/ #
# Tensorflow: https://www.tensorflow.org/tutorials/         #
#############################################################

from mfcc import mfcc
from mfcc import delta
from mfcc import log_filter_bank
import scipy.io.wavfile as wav

import tensorflow as tf

handle = open('./dataset.csv', 'r')
raw_lines = handle.read().split('\n')
handle.close()
lines = [(lambda x: x.split(','))(line) for line in raw_lines]
print(lines)

sess = tf.Session()

#lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

for filename, target_emotion in lines:
    rate, signal = wav.read(filename)
    mfcc_feature = mfcc(signal, rate)
    d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)
    print(filter_bank_feature[1:3, :])

EMOTIONS = ["happiness", "sadness", "neutral", "rage", "nervous"]