#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

gender_df = pd.read_csv('data/gender_submission.csv')
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print("Train data frame shape is", train_df.shape)
print("Test data frame shape is", test_df.shape)

print("Train columns:", train_df.columns)
print("Train head(10)")
print(train_df.head(10))

# getting the summary statistics of the numerical columns #
train_df.describe()

# let us get some plots to see the data #
train_df.Survived.value_counts().plot(kind='bar', alpha=0.6)
plt.title("Distribution of Survival, (1 = Survived)")
plt.show()