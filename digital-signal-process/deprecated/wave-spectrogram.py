from scipy.io import wavfile as wav
from scipy import signal
import matplotlib.pyplot as plt

sample_rate, sample_signal = wav.read('./sample_24414.wav')

freq, times, spectrogram = signal.spectrogram(sample_signal, sample_rate)

print('spectrogram:', spectrogram)

fig = plt.figure()

plt.pcolormesh(times, freq, spectrogram)
plt.imshow(spectrogram)
plt.axis('off')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
fig.savefig('./pyplot_figure_image.png')
#plt.close(fig)