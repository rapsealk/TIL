#!/usr/bin/env python3
"""
코드 6-2: 문자 수준의 원-핫 인코딩
"""
import string
import numpy as np

samples = ['The cat sat on mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters)+1)))

max_length = 50

results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

print('token_index:', token_index)
print('results:', results)