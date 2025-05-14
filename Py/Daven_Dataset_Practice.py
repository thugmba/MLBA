#
# Dataset Practice (Smoker)
# Daven
# 2025/5/5
#

import pandas as pd
import matplotlib.pyplot as plt

# 1. Input
df = pd.read_csv('smoker.csv')


# 2. Process 
print(df.shape)
print(df.info())
print(df.head())
print(df.tail())

print(df['smoker'].mean())
print(df['treatment'].mean())
print(df['outcome'].mean())

print(df.sum(axis = 1))
print(df.sum(axis = 0))

print(df.describe())

# 3. Output 
df['smoker'].hist()
plt.show()