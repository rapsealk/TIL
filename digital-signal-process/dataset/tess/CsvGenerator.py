#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

#################################################################
# set current directory to '/dataset/tess' before run.      #
#################################################################

class CsvGenerator:

    def __init__(self):
        print('CsvGenerator initialized.')

    def save_as_csv(self, emotions):
        self.csv_file = open('dataset.csv', 'w')
        for emotion in emotions:
            files = os.listdir(os.getcwd()+'\\'+emotion)
            for filename in files:
                self.csv_file.write(filename + ',' + emotion + '\n')
        self.csv_file.close()

if __name__ == '__main__':
    csv_generator = CsvGenerator()
    emotions = ['happy', 'neutral', 'angry', 'sad', 'disgust']
    csv_generator.save_as_csv(emotions)