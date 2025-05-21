import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_percentage_error
from polars.selectors import numeric
import joblib

# Load the CSV data
data = pl.read_csv(
    r'D:\Python Projects\Learning\Machine Learning\Solving problem\Spendings\Percent_Change_in_Consumer_Spending.csv'
)

# Drop target column and extract date
X = data.drop("total spending")
y = data["total spending"]
date = X["Date"]

# Select numeric features
X_numeric = X.select(numeric())

# Scale the features before applying One-Class SVM
scaler = StandardScaler()
X_scaled_np = scaler.fit_transform(X_numeric.to_numpy())
OCSVM = OneClassSVM(nu=0.05, gamma=0.25)
OCSVM.fit(X_scaled_np)
pred = OCSVM.predict(X_scaled_np)
mask = pred == 1  # +1 = inlier, -1 = outlier

# Filter data based on inliers
refined_X_np = X_scaled_np[mask]
refined_y = y.filter(mask)
refined_date = date.filter(mask)

# Convert scaled NumPy array back to Polars DataFrame with original column names
refined_X = pl.DataFrame(refined_X_np, schema=X_numeric.columns)
final_X = refined_X.with_columns(Date=refined_date)

# Setup time series cross-validation
n_samples = len(final_X)
ts_cv = TimeSeriesSplit(
    n_splits=3,
    gap=360,
    max_train_size=int(0.75 * n_samples),
    test_size=int(0.25 * n_samples)
)

# Get the first split
all_splits = list(ts_cv.split(refined_X_np, refined_y))
train_idx, test_idx = all_splits[0]

X_train, X_test = refined_X_np[train_idx], refined_X_np[test_idx]
y_train, y_test = refined_y[train_idx], refined_y[test_idx]

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, criterion='squared_error', min_samples_split=2)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_preds)
print(f"Random Forest MAPE: {rf_mape:.4f}")

# Train Linear Regression for comparison
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_mape = mean_absolute_percentage_error(y_test, lr_preds)
print(f"Linear Regression MAPE: {lr_mape:.4f}")

# Choose the better model
if rf_mape < lr_mape:
    best_model = rf_model
    model_name = "RandomForest"
else:
    best_model = lr_model
    model_name = "LinearRegression"

# Save the model
joblib.dump(best_model, f"{model_name}_best_model.pkl")
print(f"Saved best model ({model_name}) to '{model_name}_best_model.pkl'")
