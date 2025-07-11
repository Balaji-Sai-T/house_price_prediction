import pandas as pd

# Column names based on the UCI description
columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
    "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# Load the CSV with no header
df = pd.read_csv("data/housing.csv", header=None, names=columns)
