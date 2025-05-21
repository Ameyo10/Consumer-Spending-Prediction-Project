---

## 🧠 Predicting Consumer Spending from State-wise Economic Data

This project builds a machine learning pipeline to predict the **total spending** of consumers based on state-wise historical economic data. It includes **data visualization**, **outlier handling**, **model tuning**, and **final prediction** using the best model.

---

## 📁 Project Structure

```plaintext
├── Data_Visualisation.py              # Visualizes trends in consumer spending
├── Tuning_model.py                   # Handles outlier detection, model training & selection
├── Prediction.py                     # Loads the trained model and predicts on test data
├── Percent_Change_in_Consumer_Spending.csv   # Main dataset used for training and visualization
├── model_test_results.csv            # Final test data with actual vs predicted spending
├── LinearRegression_best_model.pkl   # Trained best model (generated at runtime)
└── README.md                         # Project documentation
```

---

## 📊 Data Description

* **Percent\_Change\_in\_Consumer\_Spending.csv** contains the percent change in spending across various sectors (like food, healthcare, entertainment) for different states over time.
* A new column, **`total spending`**, is computed by summing spending across all major sectors.

---

## ⚙️ How the Pipeline Works

### 1. `Data_Visualisation.py`

* Groups and plots monthly spending trends across different categories.
* Helps identify key patterns, such as peaks, declines, or seasonal fluctuations.

### 2. `Tuning_model.py`

* **Selects numeric features** and removes outliers using `OneClassSVM`.
* Applies **feature scaling** using `StandardScaler`.
* Splits the dataset using **TimeSeriesSplit** with a gap to avoid leakage.
* Compares `RandomForestRegressor` and `LinearRegression` using **MAPE**.
* Saves the **best-performing model** (`.pkl` file) for later use.

### 3. `Prediction.py`

* Loads `model_test_results.csv`, which contains features, actual spending, and original model predictions.
* Reloads the trained model (`.pkl`).
* Recomputes predictions and compares them with the actual values using **MAPE**.

---

## ✅ How to Use

### 🔧 1. Prepare the Environment

```bash
pip install pandas scikit-learn matplotlib joblib
```

### 📈 2. Visualize the Data

```bash
python Data_Visualisation.py
```

### 🤖 3. Train and Tune Models

```bash
python Tuning_model.py
```

* This will generate and save `RandomForest_best_model.pkl` or `LinearRegression_best_model.pkl`.

### 🧪 4. Predict and Evaluate

```bash
python Prediction.py
```

---

## 📄 Output

* **`model_test_results.csv`** contains:

  * Input features
  * Date
  * Actual total spending
  * Predicted spending
* You can use this file to evaluate your model or visualize accuracy.

---

## 📝 Future Improvements

* Add support for more models (e.g., XGBoost, SVR)
* Perform hyperparameter optimization using `GridSearchCV`
* Incorporate external economic indicators or news sentiment
* Add model interpretability tools (e.g., SHAP)

---

## 🧠 Author

*Project developed by Ameyo Jha as part of a consumer spending prediction analysis pipeline.*

---
