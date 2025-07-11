import pandas as pd

# Load data
df = pd.read_csv("data/housing.csv")

# Basic info
print("\n🔹 Dataset Shape:", df.shape)
print("\n🔹 First 5 Rows:\n", df.head())
print("\n🔹 Summary Statistics:\n", df.describe())
print("\n🔹 Missing Values:\n", df.isnull().sum())
print("\n🔹 Data Types:\n", df.dtypes)
