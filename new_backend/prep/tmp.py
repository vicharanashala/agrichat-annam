import pandas as pd
import os

df = pd.read_csv('data/kcc_dataset_part_1.csv')

# Drop row nos which don't have row no. % 10 == 1. Row no. is not a field in
# the dataset, but it is the index of the dataframe.
df = df[df.index % 10 == 1]
# Save the filtered dataframe to a new CSV file
df.to_csv('data/kcc_dataset_part_1_filtered.csv', index=False)

