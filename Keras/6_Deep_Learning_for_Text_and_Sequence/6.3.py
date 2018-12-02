#!/usr/bin/env python3
"""
코드 6-3: 케라스를 사용한 단어 수준의 원-핫 인코딩하기
"""
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print('one_hot_results:', one_hot_results)
print('%s개의 고유한 토큰을 찾았습니다.' % len(word_index))