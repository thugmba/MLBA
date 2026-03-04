#
# Read & Modify CSV
# 03/26/2025
#

import pandas as pd

# 1. Input
raw_data = pd.read_csv("Menu.csv")
print(raw_data)
print(raw_data.info())

# 2. Process
total = raw_data["Price"].sum()
print(f"Var: {round(raw_data['Price'].var(), 2)}")
print(f"Std: {round(raw_data['Price'].std(), 2)}")

# 3. Output
print(f"Total: {total}")

# 4. Extra
print(len(raw_data.index))
raw_data.loc[len(raw_data.index)] = ["Total", total]

raw_data.to_csv("Menu_total.csv")