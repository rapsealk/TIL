from scipy.io import wavfile as wav
from scipy import signal
# pip install Pillow
from scipy.misc import imsave
import matplotlib.pyplot as plt

sample_rate, sample_signal = wav.read('./sample_24414.wav')

freq, times, spectrogram = signal.spectrogram(sample_signal, sample_rate)

print('spectrogram:', spectrogram)

"""
plt.pcolormesh(times, freq, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
"""

imsave('sample_image.png', spectrogram)