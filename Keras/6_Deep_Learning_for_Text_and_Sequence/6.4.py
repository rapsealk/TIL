#!/usr/bin/env python3
"""
코드 6-4: 해싱 기법을 사용한 단어 수준의 원-핫 인코딩하기
"""
samples = ['The cat sat on mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.