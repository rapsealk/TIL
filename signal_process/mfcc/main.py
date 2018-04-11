#!/usr/bin/python3
# -*- coding: utf-8 -*-

######################################################################
# Reference: http://python-speech-features.readthedocs.io/en/latest/ #
######################################################################

from mfcc import mfcc
from mfcc import delta
from mfcc import log_filter_bank
import scipy.io.wavfile as wav

if __name__ == '__main__':
    filename = input('filename: ')
    rate, signal = wav.read(filename)
    mfcc_feature = mfcc(signal, rate)
    d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)

    print(filter_bank_feature[1:3, :])