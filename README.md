# 🏡 House Price Prediction

## Project Overview
The goal of this project is to predict housing prices based on various features using machine learning algorithms such as **Linear Regression, Ridge, Lasso, Decision Tree**, and **XGBoost**. By analyzing factors like income, house age, and geographical proximity, this model can assist in estimating housing values — useful for real estate investors and analysts.

## Dataset Description
The dataset used is the California Housing Dataset, which includes demographic and geographic features from various districts.

### **Key Features in the Dataset:**
1. **MedInc** – Median income in block group.
2. **HouseAge** – Median house age in block group.
3. **AveRooms** – Average number of rooms per household.
4. **AveBedrms** – Average number of bedrooms per household.
5. **Population** – Population in the block group.
6. **AveOccup** – Average household occupancy.
7. **Latitude** – Latitude coordinate of block group.
8. **Longitude** – Longitude coordinate of block group.
9. **Ocean Proximity** – Category of how close the location is to the ocean.
10. **MedianHouseValue** – The target variable to predict.

## Machine Learning Algorithms Used

### **1. Linear Regression**
- A baseline model that assumes a linear relationship between features and the target.
- Easy to interpret and fast to train.

### **2. Ridge Regression**
- A regularized version of linear regression.
- Helps reduce overfitting by penalizing large coefficients (L2 regularization).

### **3. Lasso Regression**
- Another regularization technique that can zero out irrelevant features (L1 regularization).
- Helps with feature selection.

### **4. Decision Tree Regressor**
- A non-linear model that splits the data into branches based on feature values.
- Captures interactions between features but prone to overfitting.

### **5. XGBoost Regressor**
- A powerful ensemble technique using gradient boosting.
- Outperforms other models in both accuracy and robustness.

## Model Evaluation
Each model was evaluated based on:
- **Root Mean Squared Error (RMSE)** – Measures the average prediction error magnitude.
- **Mean Absolute Error (MAE)** – The average of absolute differences between predictions and actual values.
- **R² Score** – Explains how much variability is captured by the model.
- **Cross-Validation Score** – Evaluates model consistency across folds.

## Results & Conclusion

| Model             | RMSE | MAE  | R² Score | CV Score |
|-------------------|------|------|----------|----------|
| Linear Regression | 4.40 | 3.19 | 0.65     | 0.36     |
| Ridge Regression  | 4.40 | 3.18 | 0.65     | 0.37     |
| Lasso Regression  | 4.44 | 3.20 | 0.65     | 0.41     |
| Decision Tree     | 4.49 | 2.48 | 0.64     | 0.13     |
| **XGBoost**       | **2.77** | **1.95** | **0.86** | **0.56** |

✅ **Final Model**: XGBoost (Best performance and saved for inference)

## Visualizations Included
- 📈 **Distribution of Target Variable**
- 📊 **Feature Correlation Heatmap**
- 📉 **Actual vs Predicted Plot**
- ⚡ **Residuals Plot**
- 🌐 **Interactive Plot (LSTAT vs MEDV) using Plotly**

All plots are saved in the `images/` folder for reference.

## Installation & Usage

### **1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/house_price_prediction.git
cd house_price_prediction
