#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Part 1 - Preprocess data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
dataset = pd.read_csv('dataset/Churn_Modelling.csv')	# DataFrame
X = dataset.iloc[:, 3:13].values	# numpy.ndarray
y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split the dataset into the Training and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Part 2 - Build ANN
"""
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Add the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))   # softmax for more classes.

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])	# categorical_crossentropy for more classes.

# Fit the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size=10, epochs=100)

"""
Part 3 - Make the predictions and evaluate the model
"""
y_pred = classifier.predict(X_test)
print('prediction:', y_pred)

y_pred = (y_pred > 0.5)
print('True or False:', y_pred)

"""
Homework Challenge - Should we say goodbye to that customer?
"""
"""
customer = [
	600,		# Credit Score
	'France',	# Geography	0, 0
	'Male',		# Gender
	40,			# Age
	3,			# Tenure
	60000,		# Balance
	2,			# Number of Products
	1,			# Has Credit Card (True)
	1,			# Is an Active Member (True)
	50000		# Estimated Salary
]
customer = np.array([customer])
customer[:, 1] = labelencoder_X_1.fit_transform(customer[:, 1])
customer[:, 2] = labelencoder_X_2.fit_transform(customer[:, 2])
customer = customer.astype(dtype=np.float64)
customer = sc.fit_transform(customer)

new_prediction = classifier.predict(customer) > 0.5
print('customer:', new_prediction)
"""
# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix:', cm)