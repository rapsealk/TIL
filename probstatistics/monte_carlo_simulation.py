#!/usr/bin/python3
# -*- coding: utf-8 -*-
from random import random

epoch = 10000
es = 0.0
while epoch > 0:
	count = 0
	summ = 0
	while summ < 1:
		summ += random()
		count += 1
	es += count
	epoch -= 1
es /= 10000
print('es: ', es)
