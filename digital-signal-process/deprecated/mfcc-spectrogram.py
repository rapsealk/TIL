#import tensorflow as tf
import numpy as np
import time

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from scipy import signal as sig
import matplotlib.pyplot as plt

from random import shuffle

if __name__ == '__main__':
    filename = './sample_24414.wav'
    rate, signal = wav.read(filename)
    mfcc_feature = mfcc(signal, rate)
    d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)
    feature = filter_bank_feature[1:3, :]

    print('signal:', signal)
    print('rate:', rate)
    print('np.shape(signal):', np.shape(signal))
    print('mfcc_feature:', mfcc_feature)
    print('np.shape(mfcc_feature):', np.shape(mfcc_feature))
    print('d_mfcc_feature:', d_mfcc_feature)
    print('np.shape(d_mfcc_feature):', np.shape(d_mfcc_feature))
    print('feature:', feature)
    print('np.shape(feature):', np.shape(feature))

    # exit(0)

    frequencies, times, spectrogram = sig.spectrogram(feature, rate)
    # frequencies, times, spectrogram = signal.spectrogram(signal, rate)

    fig = plt.figure()

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.axis('off')
    fig.savefig('./a.png')