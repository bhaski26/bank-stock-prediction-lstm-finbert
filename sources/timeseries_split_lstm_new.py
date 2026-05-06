import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score,
accuracy_score,
precision_score,
recall_score,
f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

# ==========================================
# LOAD DATA
# ==========================================

df=pd.read_csv(
r"C:/Users/bhask/Downloads/Minor Project 2025-26/Datasets/final_stock_sentiment_dataset.csv"
)

df.columns=df.columns.str.lower()

df.rename(
columns={
"unnamed: 1":"open",
"unnamed: 2":"high",
"unnamed: 3":"low",
"unnamed: 4":"close",
"unnamed: 5":"volume"
},
inplace=True
)

df["date"]=pd.to_datetime(
df["date"],
errors="coerce"
)

df=df.sort_values(
"date"
)

features=[
"open",
"high",
"low",
"close",
"volume",
"daily_sentiment"
]

df=df.dropna(
subset=features
)

data=df[
features
].values

print("\nDataset Loaded")
print(
"Rows:",
len(df)
)

# ==========================================
# NORMALIZE
# ==========================================

scaler=MinMaxScaler()

data_scaled=scaler.fit_transform(
data
)

# save scaler
joblib.dump(
scaler,
"scaler.save"
)

# ==========================================
# CREATE SEQUENCES
# ==========================================

X=[]
y=[]

sequence_length=60

for i in range(
sequence_length,
len(data_scaled)
):

    X.append(
        data_scaled[
            i-sequence_length:i
        ]
    )

    y.append(
        data_scaled[i,3]
    )

X=np.array(X)
y=np.array(y)

print(
"\nSequence Shape:",
X.shape
)

# ==========================================
# TIME SERIES SPLIT
# ==========================================

tscv=TimeSeriesSplit(
n_splits=5
)

mae_scores=[]
mse_scores=[]
rmse_scores=[]
r2_scores=[]

acc_scores=[]
prec_scores=[]
rec_scores=[]
f1_scores=[]

# ==========================================
# CROSS VALIDATION
# ==========================================

for fold,(train_index,test_index) in enumerate(
tscv.split(X)
):

    print(
    f"\n========== Fold {fold+1} =========="
    )

    X_train=X[
        train_index
    ]

    X_test=X[
        test_index
    ]

    y_train=y[
        train_index
    ]

    y_test=y[
        test_index
    ]

    print(
    "Train Shape:",
    X_train.shape
    )

    print(
    "Test Shape:",
    X_test.shape
    )

    # ======================================
    # MODEL
    # ======================================

    model=Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                X.shape[1],
                X.shape[2]
            )
        )
    )

    model.add(
        LSTM(32)
    )

    model.add(
        Dense(1)
    )

    # fixed compile
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # ======================================
    # TRAIN
    # ======================================

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    # ======================================
    # PREDICT
    # ======================================

    predictions=model.predict(
        X_test
    )

    # ======================================
    # REGRESSION METRICS
    # ======================================

    mae=mean_absolute_error(
        y_test,
        predictions
    )

    mse=mean_squared_error(
        y_test,
        predictions
    )

    rmse=np.sqrt(
        mse
    )

    r2=r2_score(
        y_test,
        predictions
    )

    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    print(
    "MAE:",
    mae
    )

    print(
    "MSE:",
    mse
    )

    print(
    "RMSE:",
    rmse
    )

    print(
    "R2:",
    r2
    )

    # ======================================
    # TREND DIRECTION METRICS
    # ======================================

    y_dir=(
        y_test[1:]>
        y_test[:-1]
    ).astype(int)

    pred_dir=(
        predictions[1:]>
        predictions[:-1]
    ).astype(int)

    acc=accuracy_score(
        y_dir,
        pred_dir
    )

    prec=precision_score(
        y_dir,
        pred_dir
    )

    rec=recall_score(
        y_dir,
        pred_dir
    )

    f1=f1_score(
        y_dir,
        pred_dir
    )

    acc_scores.append(
        acc
    )

    prec_scores.append(
        prec
    )

    rec_scores.append(
        rec
    )

    f1_scores.append(
        f1
    )

    print(
    "Accuracy:",
    acc
    )

    print(
    "Precision:",
    prec
    )

    print(
    "Recall:",
    rec
    )

    print(
    "F1:",
    f1
    )

# ==========================================
# FINAL CROSS VALIDATION RESULTS
# ==========================================

print(
"\n========== Average Results =========="
)

print(
"Average MAE:",
np.mean(
mae_scores
)
)

print(
"Average MSE:",
np.mean(
mse_scores
)
)

print(
"Average RMSE:",
np.mean(
rmse_scores
)
)

print(
"Average R2:",
np.mean(
r2_scores
)
)

print(
"Average Accuracy:",
np.mean(
acc_scores
)
)

print(
"Average Precision:",
np.mean(
prec_scores
)
)

print(
"Average Recall:",
np.mean(
rec_scores
)
)

print(
"Average F1:",
np.mean(
f1_scores
)
)

# ==========================================
# FINAL MODEL TRAINING ON FULL DATA
# ==========================================

print(
"\nTraining final model..."
)

final_model=Sequential()

final_model.add(
LSTM(
64,
return_sequences=True,
input_shape=(
X.shape[1],
X.shape[2]
)
)
)

final_model.add(
LSTM(32)
)

final_model.add(
Dense(1)
)

final_model.compile(
optimizer="adam",
loss="mean_squared_error"
)

final_model.fit(
X,
y,
epochs=20,
batch_size=32
)

# ==========================================
# SAVE MODEL (.keras)
# ==========================================

final_model.save(
"bank_lstm_model.keras"
)

print(
"\nModel saved as bank_lstm_model.keras"
)

print(
"Scaler saved as scaler.save"
)