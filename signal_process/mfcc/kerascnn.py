import numpy
import scipy.io.wavfile as wav
from mfcc import mfcc
from mfcc import delta
from mfcc import log_filter_bank
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils


""" if __name__ == '__main__':
    filename = 'english.wav'
    rate, signal = wav.read(filename)
    mfcc_feature = mfcc(signal, rate)
    d_mfcc_feature = delta(mfcc_feature, 2)
    filter_bank_feature = log_filter_bank(signal, rate) """

width = 28
height = 28

#데이터셋 생성
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, width, height, 1).astype('float32')/255
x_test = x_test.reshape(10000, width, height, 1).astype('float32')/255

x_val = x_train[50000:]
y_val = y_train[50000:]
x_train = x_train[:50000]
y_train = y_train[:50000]

y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

#CNN 신경망 모델링
#컨볼루션 - 컨볼루션 - 맥스풀링 레이어가 2번 반복. 0.25의 확률로 드랍아웃되며
#1번째에는 32개의 뉴런을, 2번째에는 64개의 뉴런을 갖는다.
#필터는 3x3이며 풀링 사이즈는 2x2
#전결합층에서는 256개의 뉴런을 가진다.
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(width,height,1)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#CNN 모델 학습 설계
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#모델 학습
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_data=(x_val, y_val))

#모델 평가
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('[loss, metrics] = %s' %(loss_and_metrics))

#모델 사용 
yhat_test = model.predict(x_test, batch_size=32)