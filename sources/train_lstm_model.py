import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# =============================
# Load Dataset
# =============================

df = pd.read_csv("datasets/final_stock_sentiment_dataset.csv")

df.columns = df.columns.str.lower()

print("\nColumns in dataset:")
print(df.columns)

df.rename(columns={
    "unnamed: 1": "open",
    "unnamed: 2": "high",
    "unnamed: 3": "low",
    "unnamed: 4": "close",
    "unnamed: 5": "volume"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.sort_values("date")

# =============================
# Feature Selection
# =============================

features = ["open","high","low","close","volume","daily_sentiment"]

df = df.dropna(subset=features)

data = df[features].values

# =============================
# Normalize Data
# =============================

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

# =============================
# Create Time Sequences
# =============================

X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i])
    y.append(data_scaled[i,3])

X = np.array(X)
y = np.array(y)

# =============================
# Train/Test Split
# =============================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =============================
# Build LSTM Model
# =============================

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# =============================
# Train Model
# =============================

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# =============================
# Predictions
# =============================

predictions = model.predict(X_test)

# =============================
# Evaluation Metrics
# =============================

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

accuracy = (1 - rmse) * 100

print("\nModel Performance Metrics")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Model Accuracy:", accuracy, "%")

# =============================
# Visualization 1
# Training vs Validation Loss
# =============================

plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

# =============================
# Visualization 2
# Actual vs Predicted Prices
# =============================

plt.figure(figsize=(12,6))

plt.plot(y_test, label="Actual Price")
plt.plot(predictions, label="Predicted Price")

plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Normalized Price")

plt.legend()

plt.show()

# =============================
# Visualization 3
# Error Distribution
# =============================

errors = y_test - predictions.flatten()

plt.figure(figsize=(10,5))

plt.hist(errors, bins=50)

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.show()

# =============================
# Visualization 4
# Residual Plot
# =============================

plt.figure(figsize=(10,5))

plt.scatter(range(len(errors)), errors)

plt.axhline(y=0)

plt.title("Residual Plot")
plt.xlabel("Samples")
plt.ylabel("Prediction Error")

plt.show()