#!/usr/bin/python3
# -*- coding: utf-8 -*-
from random import random

# Generate 10000 random numbers
random_numbers = [random() for i in range(10000)]
print(random_numbers)

# Calculate mean
mean = sum(random_numbers) / 10000
print('mean: ', mean)

# Calculate variance
square_mean = sum([random_number ** 2 for random_number in random_numbers]) / 10000
print('square_mean: ', square_mean)
variance = square_mean - (mean ** 2)
print('variance: ', variance)

a = 0
b = 1
population_mean = float(b - a) / 2
population_variance = (float(b - a) ** 2) / 12
print('population_mean: ', population_mean)
print('population_variance: ', population_variance)
