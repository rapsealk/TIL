#!/usr/bin/python3
# -*- coding: utf-8 -*-

#############################################################
# Reference:  https://www.svds.com/tensorflow-rnn-tutorial/ #
# Tensorflow: https://www.tensorflow.org/tutorials/         #
#############################################################

from mfcc import mfcc
from mfcc import delta
from mfcc import log_filter_bank
import scipy.io.wavfile as wav

#import tensorflow as tf
import numpy as np
import scipy.special

# 신경망 클래스의 정의
class CustomNetwork:

    # 신경망 초기화
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정
        self.input_nodes = input_nodes
        # TODO("#1 multiple hidden-layers")
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # 학습률
        self.learning_rate = learning_rate
        # 가중치 행렬 wih, who
        # 배열 내 가중치는 w_i_j로 표기. 노드 i에서 다음 계층의 노드 j로 연결됨을 의미
        self.weight_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weight_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        
        # 활성화 함수로는 시그모이드 함수를 이용
        self.activation_function = lambda x: scipy.special.expit(x)
        # self.backward_query_function = lambda x: scipy.special.logit(x)

        #print("initialized ---")
        #print("weight_input_hidden")
        #print(self.weight_input_hidden)
        #print("weight_hidden_output")
        #print(self.weight_hidden_output)

        pass

    # 신경망 학습시키기
    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # 은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)

        # 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs
        # 은닉 계층의 오차는 가주치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)
        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.weight_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # 입력 계층과 은닉 계층 간의 가중치 업데이트
        self.weight_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        #print("trained ---")
        #print("weight_input_hidden")
        #print(self.weight_input_hidden)
        #print("weight_hidden_output")
        #print(self.weight_hidden_output)
        pass

    # 신경망에 질의하기
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = np.array(inputs_list, ndmin=2).T
        # 은닉 계층으로 들어오는 신호를 계산
        #print('weight_input_hidden.shape:', self.weight_input_hidden.shape)
        #print('inputs.shape:', inputs.shape)
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        # 은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)
        #print('final_outputs:', final_outputs)
        return final_outputs

if __name__ == '__main__':

    EMOTIONS = ["happiness", "sadness", "neutral", "anger", "nervous"]

    # 입력, 은닉, 출력 노드의 수
    input_nodes = 26 # 784
    hidden_nodes = 100
    output_nodes = 5 # 10

    # 학습률은 0.25
    learning_rate = 0.25 # 0.3

    # 신경망의 인스턴스 생성
    n = CustomNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # mnist 학습 데이터인 csv 파일을 리스트로 불러오기
    # training_data_file = open('mnist_dataset/mnist_train.csv', 'r')
    # training_data_list = training_data_file.readlines()
    # training_data_file.close()

    # 신경망 학습시키기
    # image_array = scipy.misc.imread(image_file_name, flatten=True)
    # image_data = 255.0 - image_array.reshape(784)
    # image_data = (image_data / 255.0 * 0.99) + 0.01

    handle = open('./dataset.csv', 'r')
    raw_lines = handle.read().split('\n')
    handle.close()
    lines = [(lambda x: x.split(','))(line) for line in raw_lines]
    print(lines)

    epoch = 100
    for e in range(epoch):
        for filename, target_emotion in lines:
            rate, signal = wav.read(filename)
            mfcc_feature = mfcc(signal, rate)
            d_mfcc_feature = delta(mfcc_feature, 2)
            filter_bank_feature = log_filter_bank(signal, rate)
            print(filter_bank_feature[1:3, :])

            inputs = np.asfarray(filter_bank_feature[1:3, :])
            targets = np.zeros(output_nodes) + 0.01
            targets[EMOTIONS.index(target_emotion)] = 0.99
            #print('inputs:', inputs)
            print('targets:', targets)
            n.train(inputs, targets)
    
    # 신경망 테스트하기

    # 신경망의 성능의 지표가 되는 성적표를 아무 값도 가지지 않도록 초기화
    scorecard = []

    for filename, emotion in lines:
        rate, signal = wav.read(filename)
        mfcc_feature = mfcc(signal, rate)
        d_mfcc_feature = delta(mfcc_feature, 2)
        filter_bank_feature = log_filter_bank(signal, rate)
        print('ideal:', emotion)
        inputs = np.asfarray(filter_bank_feature[1:3, :])
        outputs = n.query(inputs)[:, :1]
        # to 1-d
        outputs = [x[0] for x in outputs]
        label = np.argmax(outputs) # - 1 # TODO("?")
        print('outputs:', outputs)
        print('label:', label)
        print('emotiosn:', EMOTIONS)
        print('actual:', EMOTIONS[label])

        scorecard.append(emotion == EMOTIONS[label])
        print('result:', emotion == EMOTIONS[label])

    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    print('performance:', performance)