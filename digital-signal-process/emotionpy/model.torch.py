#!/usr/bin/env python3
import torch
import torchvision
import torch.nn as nn
from torch import autograd
from keras.utils import np_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from speechpy import mfcc
from speechpy import delta
from speechpy import log_filter_bank
from speechpy import preprocessor
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt

from random import shuffle

def to_categorical(y, num_classes):
    # one-hot encodes a tensor
    return np.eye(num_classes, dtype='uint8')[y]

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size()[0], -1)


class CrossEntropyLoss(nn.Module):

    log_softmax = nn.LogSoftmax()

    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).cuda())

    def forward(self, logits, target):
        log_probabilities = self.log_softmax(logits)
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()


class Model(object):    # nn.Module

    def __init__(self):
        self.EMOTIONS = { 'happy': 0, 'neutral': 1, 'sad': 2, 'angry': 3 }

		self.num_sample = 32    # 60
        self.n_mfcc 26
        self.num_channels = 1
        self.batch_size = 32
		self.epochs = 20

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(2, 2)),
            nn.Softmax2d(),
            nn.MaxPool2d(kernel_size=(4, 4)),
            nn.Flatten(),

            nn.Linear(out_features=128),
            nn.ReLU(),
            nn.Linear(out_features=len(self.EMOTIONS)),
            nn.Sigmoid()
        )
        self.model = self.model.to(device)

        self.loss = CrossEntropyLoss(len(self.EMOTIONS))
        self.optimizer = torch.optim.Adam(self.model.parameters())  # lr=1e-3

    def load_dataset(self, csv='./dataset/dataset.csv'):
        handle = open(csv, 'r')
        rawlines = handle.read().split('\n')
        handle.close()
        datalist = [line.split(',') for line in rawlines]
        shuffle(datalist)
        cnt = 0
        processed_x = []
        processed_y = []
        for filename, target_emotion in datalist:
            # 'YAF_germ_angry.wav': continue
            rate, signal = wav.read('./dataset/tess/{}/{}'.format(target, filename))
            filter_bank_feature = log_filter_bank(signal, rate)
            processed_x.append(filter_bank_feature)
            processed_y.append(self.EMOTIONS[target_emotion])
        processed_x = np.array(processed_x).reshape(-1, self.num_sample, self.n_mfcc, 1)
        processed_y = np_utils.to_categorical(processed_y)

        ratio = 0.9
        n_train = int(len(datalist) * ratio)

        self.train_x = processed_x[:n_train]
        self.train_y = processed_y[:n_train]

        self.test_x = processed_x[n_train:]
        self.test_y = processed_y[n_train:]

    def train(self):
        self.learning_rate = 1e-3
        #self.model.train()
        for i in range(self.epochs):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(self.train_x[i])
            # Compute loss
            loss = self.loss(y_pred, self.train_y[i])
            print(i, loss.item())
            # Zero the gradients before running the backward pass.
            self.model.zero_grad()
            # Backward pass: compute gradient of the loss
            loss.backward()

            with torch.no_grad():
                for param in self.model.parameters():
                    param -= self.learning_rate * param.grad