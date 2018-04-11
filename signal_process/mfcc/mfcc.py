import numpy as np
from scipy.fftpack import dct
import signal_process

def mfcc(signal, sample_rate=16000, window_length=0.025, window_step=0.01, number_of_cepstra=13,
        number_of_filter=26, number_of_fft=512, lowest_freq=0, highest_freq=None, preemphasis=0.97,
        cepstra_lifter=22, appendEnergy=True, window_function=lambda x: np.ones((x,))):

    feature, energy = filter_bank(signal, sample_rate, window_length, window_step, number_of_filter, number_of_fft,
                                    lowest_freq, highest_freq, preemphasis, window_function)
    feature = np.log(feature)
    feature = dct(feature, type=2, axis=1, norm='ortho')[:, :number_of_cepstra]
    feature = lifter(feature, cepstra_lifter)

    if appendEnergy: feature[:,0] = np.log(energy)

    return feature

def filter_bank(signal, sample_rate=16000, window_length=0.025, window_step=0.01,
                number_of_filters=26, number_of_fft=512, lowest_freq=0, highest_freq=None,
                preemphasis=0.97, window_function=lambda x: np.ones((x,))):

    highest_freq = highest_freq or (sample_rate / 2)
    signal = signal_process.pre_emphasis(signal, preemphasis)
    frames = signal_process.frame_signal(signal, window_length * sample_rate, window_step * sample_rate, window_function)
    power_spec = signal_process.power_spectrum(frames, number_of_fft)
    energy = np.sum(power_spec, 1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)

    fbank = get_filterbanks(number_of_filters, number_of_fft, sample_rate, lowest_freq, highest_freq)
    feature = np.dot(power_spec, fbank.T)
    feature = np.where(feature == 0, np.finfo(float).eps, feature)

    return feature, energy

def log_filter_bank(signal, sample_rate=16000, window_length=0.025, window_step=0.01,
                    number_of_filters=26, number_of_fft=512, lowest_freq=0, highest_freq=None, preemphasis=0.97):
    
    feature, energy = filter_bank(signal, sample_rate, window_length, window_step, number_of_filters, number_of_fft,
                                    lowest_freq, highest_freq, preemphasis)
    return np.log(feature)

def spectral_subband_centroid(signal, sample_rate=16000, window_length=0.025, window_step=0.01,
                                number_of_filters=26, number_of_fft=512, lowest_freq=0, highest_freq=None,
                                preemphasis=0.97, window_function=lambda x: np.ones((x,))):
    
    highest_freq = highest_freq or (sample_rate / 2)
    signal = signal_process.pre_emphasis(signal, preemphasis)
    frames = signal_process.frame_signal(signal, window_length * sample_rate, window_step * sample_rate, window_function)
    power_spec = signal_process.power_spectrum(frames, number_of_fft)
    power_spec = np.where(power_spec == 0, np.finfo(float).eps, power_spec)

    filter_bank = get_filterbanks(number_of_filters, number_of_fft, sample_rate, lowest_freq, highest_freq)
    feature = np.dot(power_spec, filter_bank.T)
    R = np.tile(np.linspace(1, sample_rate / 2, np.size(power_spec, 1)), (np.size(power_spec, 0), 1))

    return np.dot(power_spec * R, filter_bank.T) / feat

def hertz2mel(hz):
    return 2595 * np.log10(1 + hz/700.)

def mel2hertz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def get_filterbanks(number_filters=20, number_fft=512, sample_rate=16000, lowest_freq=0, highest_freq=None):
    highest_freq = highest_freq or sample_rate / 2
    assert highest_freq <= sample_rate / 2, 'highest_frequency is greater than sample_rate / 2'

    low_mel = hertz2mel(lowest_freq)
    high_mel = hertz2mel(highest_freq)
    melpoints = np.linspace(low_mel, high_mel, number_filters+2)
    binn = np.floor((number_fft + 1) * mel2hertz(melpoints) / sample_rate)

    fbank = np.zeros([number_filters, number_fft // 2 + 1])
    for i in range(number_filters):
        for j in range(int(binn[i]), int(binn[i+1])):
            fbank[i, j] = (j - binn[i]) / (binn[i+1] - binn[i])
        for j in range(int(binn[i+1]), int(binn[i+2])):
            fbank[i, j] = (binn[i+2] - j) / (binn[i+2] - binn[i+1])
    
    return fbank

def lifter(cepstra, L=22):
    if L > 0:
        frames_number, number_coefficient = np.shape(cepstra)
        n = np.arange(number_coefficient)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    return cepstra

def delta(feature, N):
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    number_of_frames = len(feature)
    denominator = 2 * sum([(i ** 2) for i in range(1, N+1)])
    delta_feature = np.empty_like(feature)
    padded = np.pad(feature, ((N, N), (0, 0)), mode='edge')
    for i in range(number_of_frames):
        delta_feature[i] = np.dot(np.arange(-N, N+1), padded[i : i + 2*N + 1]) / denominator
    return delta_feature