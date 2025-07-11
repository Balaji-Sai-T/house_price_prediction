import pandas as pd
from joblib import load
import numpy as np

# Load pre-trained model and preprocessor
model = load("models/best_model.joblib")
preprocessor = load("models/preprocessor.joblib")

# Example new input data (replace with real input)
new_data = pd.DataFrame([{
    'CRIM': 0.1,
    'ZN': 18.0,
    'INDUS': 2.3,
    'CHAS': 0,
    'NOX': 0.5,
    'RM': 6.0,
    'AGE': 65.0,
    'DIS': 4.5,
    'RAD': 1,
    'TAX': 296.0,
    'PTRATIO': 15.3,
    'B': 396.9,
    'LSTAT': 4.98
}])

# Apply preprocessing
X_processed = preprocessor.transform(new_data)

# Make prediction
prediction = model.predict(X_processed)

# Output the result
print(f"🏠 Predicted House Price: ${prediction[0]*1000:.2f}")
