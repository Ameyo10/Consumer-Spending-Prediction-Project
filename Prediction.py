import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error

# Step 1: Load test data from CSV
test_data = pd.read_csv(r"D:\Python Projects\Learning\Machine Learning\Solving problem\Spendings\model_test_results.csv")

# Step 2: Extract actual values and features
y_true = test_data["Actual Spending"]
X_features = test_data.drop(columns=["Date", "Actual Spending", "Predicted Spending"])

# Step 3: Load the saved model (change filename if needed)
model = joblib.load(r"D:\Python Projects\Learning\Machine Learning\Solving problem\Spendings\LinearRegression_best_model.pkl")

# Step 4: Predict
y_pred = model.predict(X_features)

# Step 5: Compare predictions
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE on loaded model and CSV data: {mape:.4f}")

