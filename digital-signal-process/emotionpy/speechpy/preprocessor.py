from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

import os
import urllib.request

def process(path, num_sample=32):
    rate, signal = wav.read(path)
    filter_bank_feature = log_filter_bank(signal, rate)
    return filter_bank_feature[:num_sample, :]

def download_and_process(url, num_sample=32):
    filename = url.split('?')[0].split('%2F')[-1]
    path = './{}'.format(filename)
    urllib.request.urlretrieve(url, path)

    feature = process(path, num_sample)

    os.remove(path)

    return feature

def load_and_process(url, num_sample=32):
    filename = url.split('?')[0].split('%2F')[-1]
    path = './dataset/monoset/{}'.format(filename)
    return process(path, num_sample)