import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# 1. Load Data
df = pd.read_csv("final_dataset_ml_ready.csv")

# 2. Feature Selection
# We want to predict Yield (or Production) based on Inputs + Climate + Satellite
target = 'Yield_Wheat_Acre' 

# Define Features (X)

features = [
    'Fertilizer_Usage_K_Tons', 
    'Mean_NDVI', 
    'Total_Rainfall_mm', 
    'Avg_Temp_C',
    'Area_Sown_Wheat'
]

print(f"Dataset Shape: {df.shape}")
print(f"Target: {target}")
print(f"Features: {features}")

# 3. Train/Test Split
X = df[features]
y = df[target]

# 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Training

# A. Baseline Model (Simple Average) - Mandatory for Rubric
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

# B. Classical Model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# C. Advanced Model (Gradient Boosting / XGBoost style)
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# 5. Evaluation & Comparison

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f} | R2 Score: {r2:.4f}")
    return rmse, r2

print("\n--- MODEL PERFORMANCE ---")
evaluate("Baseline (Mean)", y_test, y_pred_base)
evaluate("Random Forest", y_test, y_pred_rf)
evaluate("Gradient Boosting", y_test, y_pred_gb)

# 6. Feature Importance Analysis (Mandatory for Report)
# We use the Gradient Boosting model for importance
importance = gb_model.feature_importances_
feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance: What drives Wheat Yield?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png") # Save for your Report!
print(" Feature Importance Plot saved as 'feature_importance.png'")
plt.show()

# 7. Error Analysis Plot (Actual vs Predicted)
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_gb, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Error Analysis: Actual vs Predicted ")
plt.savefig("error_analysis.png") # Save for Report
print(" Error Analysis Plot saved as 'error_analysis.png'")

# 8. Save the Best Model for PoC App
# We will use the Gradient Boosting model as our final "Product"
joblib.dump(gb_model, "final_model.pkl")
print("\n SUCCESS! Trained model saved as 'final_model.pkl'.")
