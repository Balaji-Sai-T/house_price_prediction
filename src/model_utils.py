import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_model_comparison(results):
    models = list(results.keys())

    rmse = [results[m]['RMSE'] for m in models]
    mae = [results[m]['MAE'] for m in models]
    r2 = [results[m]['R²'] for m in models]
    cv = [results[m]['CV Score'] for m in models]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("📊 Model Comparison", fontsize=18, weight='bold')

    sns.barplot(x=models, y=rmse, ax=axs[0, 0], palette="coolwarm")
    axs[0, 0].set_title("Root Mean Squared Error")
    axs[0, 0].set_ylabel("RMSE")
    axs[0, 0].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=mae, ax=axs[0, 1], palette="BuGn_d")
    axs[0, 1].set_title("Mean Absolute Error")
    axs[0, 1].set_ylabel("MAE")
    axs[0, 1].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=r2, ax=axs[1, 0], palette="Greens")
    axs[1, 0].set_title("R² Score")
    axs[1, 0].set_ylabel("R²")
    axs[1, 0].tick_params(axis='x', rotation=20)

    sns.barplot(x=models, y=cv, ax=axs[1, 1], palette="magma")
    axs[1, 1].set_title("Cross-Validation Score")
    axs[1, 1].set_ylabel("CV Score")
    axs[1, 1].tick_params(axis='x', rotation=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots top N feature importances from a tree-based model (like XGBoost).
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ This model doesn't support feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette="viridis")
    plt.title("🌟 Top Feature Importances", fontsize=16, weight='bold')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    