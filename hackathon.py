import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset

print("--- Step 1: Loading Dataset ---")

df = pd.read_csv("Delhi_Heat_Dataset_50000.csv")


# Cleaning the dataset

print("--- Step 2: Cleaning Data ---")
df.drop_duplicates(inplace=True)
df.dropna(axis=1, how='all', inplace=True)
df.dropna(subset=['LST'], inplace=True)

numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Print the dimensions of the table (Rows, Columns)
# print("Cleaned data shape:", df.shape)

# print("\n--- Cleaned Data Summary ---")
# df.info()