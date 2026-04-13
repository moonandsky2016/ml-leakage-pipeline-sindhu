# -------------------------------
# Task 1 — Create Dataset & Train Model
# -------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create synthetic dataset (50+ records)
np.random.seed(42)

n = 60
area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Create price (with some noise)
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.5 +
    np.random.normal(0, 10, n)
)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

# Features and target
X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Print intercept and coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Predictions
y_pred = model.predict(X)

# Show first 5 actual vs predicted
print("\nFirst 5 Actual vs Predicted:")
for actual, pred in list(zip(y[:5], y_pred[:5])):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")


# -------------------------------
# Task 2 — Model Evaluation
# -------------------------------

mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nEvaluation Metrics:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# Explanation:
# MAE shows the average absolute error between predicted and actual values.
# RMSE penalizes larger errors more heavily, giving insight into variance of errors.
# R² indicates how well the model explains variance (closer to 1 = better fit).


# -------------------------------
# Task 3 — Residuals & Plot
# -------------------------------

# Residuals
residuals = y - y_pred

# Histogram
plt.figure()
plt.hist(residuals, bins=15)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Explanation:
# A residual is the difference between actual and predicted value.
# If the histogram is roughly symmetric and centered around zero,
# it suggests the model errors are random and the model fits well.