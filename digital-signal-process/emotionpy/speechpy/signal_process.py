#!/usr/bin/python3
# -*- coding: utf-8 -*-

#################################################################################################################
# Reference: https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py #
#################################################################################################################

import numpy as np
import math
import decimal
import logging

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(arr, window, step=1):
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)[::step]

def frame_signal(signal, frame_length, frame_step, window_function=lambda x: np.ones((x,)), stride_trick=True):
    signal_length = len(signal)
    frame_length = int(round_half_up(frame_length))
    frame_step = int(round_half_up(frame_step))
    
    if signal_length <= frame_length:
        frames_number = 1
    else:
        frames_number = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
    
    padding_length = int((frames_number - 1) * frame_step + frame_length)
    zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((signal, zeros))

    if stride_trick:
        window = window_function(frame_length)
        frames = rolling_window(padded_signal, window=frame_length, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_length), (frames_number, 1)) + np.tile(np.arange(0, frames_number * frame_step, frame_step), (frame_length, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padded_signal[indices]
        window = np.tile(window_function(frame_length), (frames_number, 1))

    return frames * window

def deframe_signal(frames, signal_length, frame_length, frame_step, window_function=lambda x: np.ones((x,))):
    frame_length = round_half_up(frame_length)
    frame_step = round_half_up(frame_step)
    frames_number = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_length, '"frames" matrix has wrong size, 2nd dim must equal to frame_length'

    indices = np.tile(np.arange(0, frame_length), (frames_number, 1)) + np.tile(np.arange(0, frames_number * frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padding_length = (frames_number - 1) * frame_step + frame_length

    if signal_length <= 0: signal_length = padding_length

    rec_signal = np.zeros((padding_length,))
    window_correction = np.zeros((padding_length,))
    window = window_function(frame_length)

    for i in range(0, frames_number):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + window + 1e-15
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal /= window_correction

    return rec_signal[0:signal_length]

def magnitude_spectrum(frames, fft_length):
    #print('np.shape(frames)[1]:', np.shape(frames)[1])
    #print('fft_length:', fft_length)
    if np.shape(frames)[1] > fft_length:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase fft_length to avoice.',
                    np.shape(frames)[1], fft_length)
    complex_spectrum = np.fft.rfft(frames, fft_length)
    return np.absolute(complex_spectrum)

def power_spectrum(frames, fft_length):
    return 1.0 / fft_length * np.square(magnitude_spectrum(frames, fft_length))
"""
def log_power_spectrum(frames, fft_length, norm=1):
    power_spec = power_spectrum(frames, fft_length)
    power_spec[power_spec <= 1e-30] = 1e-30
    log_power_spec = 10 * np.log10(power_spec)
    if norm: log_power_spec -= np.max(log_power_spec)
    return log_power_spec
"""
def pre_emphasis(signal, coefficient=0.95):
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])