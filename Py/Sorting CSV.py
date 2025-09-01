# 
# Sorting CSV
# 03/26/2025
#

import pandas as pd

# 1. Input
raw_data = pd.read_csv("Menu.csv")
print(raw_data)
print(raw_data.info())

# 2. Process
raw_data["Price"].sort_values()

raw_data.sort_values("Price", ascending = False)

