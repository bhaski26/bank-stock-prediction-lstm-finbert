import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================
# Load Dataset
# =============================

df = pd.read_csv("C:/Users/bhask/Downloads/Minor Project 2025-26/Datasets/final_stock_sentiment_dataset.csv")

df.columns = df.columns.str.lower()

df.rename(columns={
    "unnamed: 1": "open",
    "unnamed: 2": "high",
    "unnamed: 3": "low",
    "unnamed: 4": "close",
    "unnamed: 5": "volume"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

features = ["open","high","low","close","volume","daily_sentiment"]

df = df.dropna(subset=features)

data = df[features].values

# =============================
# Normalize
# =============================

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =============================
# Create Sequences
# =============================

X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])
    y.append(data_scaled[i,3])   # close price

X = np.array(X)
y = np.array(y)

print("Total sequences:", X.shape)

# =============================
# TimeSeriesSplit
# =============================

tscv = TimeSeriesSplit(n_splits=5)

rmse_scores = []
mae_scores = []
mse_scores = []
r2_scores = []

acc_scores = []
prec_scores = []
rec_scores = []
f1_scores = []

# =============================
# Cross Validation Loop
# =============================

for fold, (train_index, test_index) in enumerate(tscv.split(X)):

    print(f"\n========== Fold {fold+1} ==========")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # =============================
    # Build Model
    # =============================

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    # =============================
    # Train
    # =============================

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # =============================
    # Predict
    # =============================

    predictions = model.predict(X_test)

    # =============================
    # Regression Metrics
    # =============================

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # =============================
    # Direction (Up/Down)
    # =============================

    y_test_dir = (y_test[1:] > y_test[:-1]).astype(int)
    pred_dir = (predictions[1:] > predictions[:-1]).astype(int)

    # =============================
    # Classification Metrics
    # =============================

    acc = accuracy_score(y_test_dir, pred_dir)
    prec = precision_score(y_test_dir, pred_dir)
    rec = recall_score(y_test_dir, pred_dir)
    f1 = f1_score(y_test_dir, pred_dir)

    acc_scores.append(acc)
    prec_scores.append(prec)
    rec_scores.append(rec)
    f1_scores.append(f1)

    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

# =============================
# Final Averages
# =============================

print("\n========== Final Average Results ==========")

print("Avg MAE :", np.mean(mae_scores))
print("Avg MSE :", np.mean(mse_scores))
print("Avg RMSE:", np.mean(rmse_scores))
print("Avg R2  :", np.mean(r2_scores))

print("Avg Accuracy :", np.mean(acc_scores))
print("Avg Precision:", np.mean(prec_scores))
print("Avg Recall   :", np.mean(rec_scores))
print("Avg F1 Score :", np.mean(f1_scores))
