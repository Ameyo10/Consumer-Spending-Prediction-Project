# 🧾 Consumer Spending Prediction Project

This project analyzes and predicts consumer spending using real-world economic data. It involves data visualization, preprocessing, model training and tuning, and finally, generating predictions on unseen data.

---

## 📁 Project Structure

├── Data_Visualisation.py # Visualizes trends and patterns in consumer spending
├── Tuning_model.py # Cleans data, performs outlier detection, tunes and selects the best model
├── Prediction.py # Loads the trained model and predicts on test data
├── Percent_Change_in_Consumer_Spending.csv # Source dataset used for training and visualization
├── model_test_results.csv # Test dataset used for evaluating predictions
├── total_spending.csv (optional) # Aggregated spending values (used in processing)
├── LinearRegression_best_model.pkl # Trained model (or RandomForest depending on tuning outcome)
---

## 🔍 Objective

Predict the **total spending** of consumers in different sectors over time using machine learning models trained on historical data.

---

## 📊 Data Visualization

- File: `Data_Visualisation.py`
- Reads from `Percent_Change_in_Consumer_Spending.csv`
- Aggregates sector-wise spending over time and by state
- Visualizes:
  - Monthly spending trends
  - Sector-wise contribution
  - Total spending distribution by State FIPS codes

---

## ⚙️ Model Tuning and Training

- File: `Tuning_model.py`
- Steps:
  1. **Outlier Removal** using One-Class SVM (on scaled numeric features)
  2. **TimeSeriesSplit** for time-aware cross-validation
  3. **Model Comparison**: Random Forest vs Linear Regression
  4. **Best Model Selection** based on Mean Absolute Percentage Error (MAPE)
  5. **Model Saving** via `joblib.dump()`

---

## 🤖 Making Predictions

- File: `Prediction.py`
- Loads `model_test_results.csv` (containing test features and actual spending)
- Loads the best model using `joblib.load()`
- Re-predicts total spending and evaluates accuracy using MAPE

---

## 📈 Dataset Info

- `Percent_Change_in_Consumer_Spending.csv`: Contains consumer spending changes across different sectors and states.
- Target Column: **`total spending`** — derived as the sum of multiple sectoral spending columns.
- Features include:
  - Date
  - State codes
  - Sectoral percent change values

---

## ✅ Requirements

- Python 3.8+
- Libraries:
  - pandas
  - matplotlib
  - scikit-learn
  - joblib

Install via:
pip install pandas matplotlib scikit-learn joblib
🚀 How to Run
Run the data visualization:

python Data_Visualisation.py
Train and save the best model:

python Tuning_model.py
Evaluate the model on test data:

python Prediction.py


All model evaluation uses MAPE for interpretability.

Code paths assume local CSV and model file access — adjust if running in a different environment.

🧠 Author
Developed by Ameyo Jha — a student passionate about data science and economic forecasting.
