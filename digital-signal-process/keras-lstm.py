import tensorflow as tf
import numpy as np
import os

from speechpy import mfcc
#from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

from random import shuffle

EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3, 'disgust': 4 }
DATA_STATIC_PATH = './dataset/tess/'

def load_dataset(csv_file=DATA_STATIC_PATH+'dataset.csv'):
    handle = open(csv_file, 'r')
    rawlines = handle.read().split('\n')
    handle.close()
    datalist = [line.split(',') for line in rawlines]

    shuffle(datalist)



    return datalist

#데이터 셔플
datalist = load_dataset()
shuffle(datalist)

train_data_ratio = 0.9
num_train_data = int(len(datalist) * train_data_ratio)
train_datalist = datalist[:num_train_data]
test_datalist = datalist[num_train_data:]

for filename, target_emotion in test_datalist:
    if filename in ['YAF_germ_angry.wav']: continue
    rate, signal = wav.read(DATA_STATIC_PATH + target_emotion + '/' + filename)
    #mfcc_feature = mfcc(signal, rate)
    #d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate)

    #print('filter_bank_feature')
    #print(filter_bank_feature)
    print(filename, np.shape(filter_bank_feature))