from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
import scipy.io.wavfile as wav

import urllib.request

def download_and_process(url):
    filename = url.split('?')[0].split('%2F')[-1]
    path = './{}'.format(filename)
    urllib.request.urlretrieve(url, path)

    num_sample = 32

    rate, signal = wav.read(path)
    filter_bank_feature = log_filter_bank(signal, rate)

    return filter_bank_feature[:num_sample, :]