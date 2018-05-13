import os
from scipy.io import wavfile as wav
from scipy import signal
# pip install Pillow
from scipy.misc import imsave
import time

EMOTIONS = ['happy', 'neutral', 'angry', 'sad', 'disgust']

if __name__ == '__main__':
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
            imsave(image_name, spectrogram)
            file_count += 1
    print('{} files - {} sec'.format(file_count, time.time() - timestamp))