import os
import argparse
import numpy as np
from scipy.io import wavfile as wav
from scipy import signal
import time

EMOTIONS = ['happy', 'neutral', 'angry', 'sad', 'disgust']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--color', type=bool)

    args = parser.parse_args()

    if args.color == True:
        import matplotlib.pyplot as plt
    else:
        # pip install Pillow
        from scipy.misc import imsave

    file_count = 0
    timestamp = time.time()
    for emotion in EMOTIONS:
        files = os.listdir(os.getcwd()+'\\'+emotion)
        for filename in files:
            if filename in ['YAF_germ_angry.wav']: continue
            wav_path = './' + emotion + '/' + filename
            sample_rate, sample_signal = wav.read(wav_path)
            freq, times, spectrogram = signal.spectrogram(sample_signal, sample_rate)
            image_name = './spectrogram/' + filename.split('.')[0] + '.png'
            spectrogram = spectrogram[:, :129] # shape(129, 129)
            if args.color == True:
                fig = plt.figure()
                plt.pcolormesh(times[:129], freq, spectrogram)
                plt.imshow(spectrogram)
                plt.axis('off')
                fig.savefig(image_name)
                plt.close(fig)
            else:
                imsave(image_name, spectrogram[:, :129])
            file_count += 1
    print('{} files - {} sec'.format(file_count, time.time() - timestamp))