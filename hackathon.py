import numpy as np
import pandas as pd
import json
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

# print(f"Dataset shape after cleaning: {df.shape}")

# print("\nMissing values after cleaning:")
# print(df.isnull().sum())



# EXTRACT SPATIAL COORDINATES

print("\n--- Step 3: Extracting Latitude & Longitude ---")

# Extract longitude and latitude from the .geo JSON column
df["Longitude"] = df[".geo"].apply(lambda x: json.loads(x)["coordinates"][0])
df["Latitude"] = df[".geo"].apply(lambda x: json.loads(x)["coordinates"][1])

# Remove the original .geo column
df.drop(columns=[".geo"], inplace=True)

print("Latitude and Longitude extracted successfully!")


# SELECT FEATURES FOR MODELLING

print("\n--- Step 4: Selecting Features ---")

input_features = [
    "NDVI",
    "NDBI",
    "Albedo",
    "Elevation",
    "Population",
    "LULC",
    "AirTemp",
    "WindSpeed",
    "Rainfall",
    "WaterDistance",
    "BuiltupDensity",
    "Latitude",
    "Longitude"
]

X = df[input_features]
y = df["LST"]

print(f"Number of Features: {len(input_features)}")


# TRAIN-TEST SPLIT

print("\n--- Step 5: Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")


# FEATURE SCALING (OPTIONAL)

print("\n--- Step 6: Feature Scaling ---")

scaler = MinMaxScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

print("Feature scaling completed!")


# SAVE FILES

print("\n--- Step 7: Saving Files ---")

# Unscaled data (recommended for Random Forest/XGBoost)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

# Scaled data (optional for future models)
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)

# Target
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n Preprocessing Completed Successfully!")
print("Files Generated:")
print("✔ X_train.csv")
print("✔ X_test.csv")
print("✔ X_train_scaled.csv")
print("✔ X_test_scaled.csv")
print("✔ y_train.csv")
print("✔ y_test.csv")