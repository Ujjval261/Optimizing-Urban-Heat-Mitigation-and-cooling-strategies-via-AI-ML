import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset

print("--- Step 1: Loading Dataset ---")

df = pd.read_csv("Delhi_Heat_Dataset_50000.csv")

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")