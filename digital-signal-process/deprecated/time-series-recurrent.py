#!/usr/bin/python3
# -*- coding: utf-8 -*-
#import tensorflow as tf
import numpy as np

def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

def toy_project(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

if __name__ == '__main__':
    print('=========================')
    print('| Time Series Recurrent |')
    print('=========================')
    T = 100
    f = toy_project(T)
    print(f)