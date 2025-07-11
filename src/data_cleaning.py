import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/housing.csv", delim_whitespace=True)
    df.columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
        "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
    ]
    return df


def get_preprocessor(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor
