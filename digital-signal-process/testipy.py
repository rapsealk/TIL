import numpy as np
import scipy.io.wavfile as wav

import os
import urllib.request

counter = { 'happy': 0, 'neutral': 0, 'sad': 0, 'angry': 0, 'disgust': 0 }

def stereo_to_mono(url, emotion='neutral', csv=None):
    global counter
    global labeller
    filename = url.split('?')[0].split('%2F')[-1]
    print(filename)
    # path = './monoset/{}'.format(filename)
    filename = '{}{}.wav'.format(emotion, counter[emotion])
    counter[emotion] += 1
    path = './monoset/{}'.format(filename)
    urllib.request.urlretrieve(url, path)

    rate, signal = wav.read(path)

    # print(signal)
    # print('<< signal')

    os.remove(path)

    mono = [(signal[i][0] + signal[i][1]) / 2 for i in range(len(signal))]
    # signal = signal.astype(float)
    # signal = signal.sum(axis=1) / 2

    mono = np.array(mono, dtype='int16')    # PCM 16 bit

    # print('<< mono')

    # sample_rate = 16000
    wav.write(path, rate, mono)

    # fhandle = open('./monoset/dataset.csv', 'w')
    csv.write('{},{}\n'.format(filename, emotion))
    # fhandle.close()


if __name__ == '__main__':
    fhandle = open('./monoset/monofy.csv', 'r')
    csv = [line.replace('\n', '').split(',') for line in fhandle.readlines()]
    fhandle.close()
    print(csv)
    labeller = open('./monoset/dataset.csv', 'w')
    for url, emotion in csv:
        stereo_to_mono(url, emotion, labeller)
    # print(csv)
    #stereo_to_mono('https://firebasestorage.googleapis.com/v0/b/kaubrain418.appspot.com/o/lsj%2F%5B%E1%84%8B%E1%85%A9%E1%84%92%E1%85%A2%E1%84%8B%E1%85%A7%E1%86%BC%5DSadandAngry.WAV?alt=media&token=9b07b036-3ba5-4579-baca-d325bb694ff1')
    labeller.close()