import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

# Local modules
from data_cleaning import load_data, get_preprocessor
from model_utils import plot_model_comparison, plot_feature_importance

# Load and preprocess data
df = load_data()
X, y, preprocessor = get_preprocessor(df)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

# Train & evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv = np.mean(cross_val_score(model, X, y, cv=5))

    results[name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "CV Score": cv
    }

    print(f"\n🔹 {name}")
    print(f"   RMSE     : {rmse:.2f}")
    print(f"   MAE      : {mae:.2f}")
    print(f"   R² Score : {r2:.2f}")
    print(f"   CV Score : {cv:.2f}")

# Select and save best model (by RMSE)
best_model_name = min(results, key=lambda k: results[k]["RMSE"])
best_model = models[best_model_name]
dump(best_model, "models/best_model.joblib")
dump(preprocessor, "models/preprocessor.joblib")

print(f"\n✅ Best model: {best_model_name} (Saved to models/)")

# Visualize Results
plot_model_comparison(results)

# Feature Importance
if hasattr(best_model, "feature_importances_"):
    feature_names = preprocessor.get_feature_names_out()
    plot_feature_importance(best_model, feature_names)
