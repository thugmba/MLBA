#
# Panda 9 Functions
# 03/26/2025
#

import pandas as pd

# 1. Input
raw_data = pd.read_csv("Menu.csv")

# 2. Process

# 3. Output
raw_data.head(3)
raw_data.tail(3)
raw_data.info()
raw_data.shape
raw_data.columns
raw_data.dtypes
raw_data.describe()
raw_data['Price'].value_counts()
raw_data.sort_values("Price", ascending = False)   