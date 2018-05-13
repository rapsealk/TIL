#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

TARGET_DIRECTORY = 'inside_out'

class Labeller:

    def __init__(self):
        print('Labeller initialized.')

    def save_as_csv(self, directory, emotions):
        self.csv_file = open(directory + '.csv', 'w')
        
        files = os.listdir(os.path.dirname(os.path.abspath(__file__)) + '\\' + directory)
        for filename in files:
            self.csv_file.write(filename + ',' + filename.split('_')[0] + '\n')

        """
        for emotion in emotions:
            files = os.listdir(os.getcwd()+'\\'+emotion)
            for filename in files:
                self.csv_file.write(filename + ',' + emotion + '\n')
        """

        self.csv_file.close()

if __name__ == '__main__':
    labeller = Labeller()
    emotions = ['happy', 'neutral', 'angry', 'sad', 'disgust']
    labeller.save_as_csv(TARGET_DIRECTORY, emotions)