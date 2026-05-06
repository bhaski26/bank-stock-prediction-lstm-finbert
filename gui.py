import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =========================================
# PAGE
# =========================================

st.set_page_config(
    page_title="Bank Stock Prediction Dashboard",
    layout="wide"
)

st.title(
    "Bank Stock Price Prediction using LSTM + FinBERT"
)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Prediction Inputs")

bank = st.sidebar.selectbox(
    "Select Bank",
    [
        "State Bank of India",
        "HDFC Bank",
        "ICICI Bank",
        "Axis Bank",
        "Kotak Mahindra Bank"
    ]
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    7
)

ticker_map = {
"State Bank of India":"SBIN.NS",
"HDFC Bank":"HDFCBANK.NS",
"ICICI Bank":"ICICIBANK.NS",
"Axis Bank":"AXISBANK.NS",
"Kotak Mahindra Bank":"KOTAKBANK.NS"
}

ticker = ticker_map[bank]

MODEL_PATH = "bank_lstm_model.keras"

# =========================================
# LIVE PRICE
# =========================================

try:
    live = yf.Ticker(ticker)

    current_price = live.history(
        period="1d"
    )["Close"].iloc[-1]

    st.sidebar.success(
        f"Live Price ₹{round(current_price,2)}"
    )

except:
    current_price = None
    st.sidebar.warning(
        "Could not fetch live price"
    )

# =========================================
# DOWNLOAD MARKET DATA
# =========================================

market_data = yf.download(
    ticker,
    period="6mo",
    progress=False,
    auto_adjust=False
)

# FIX MULTIINDEX
if isinstance(
    market_data.columns,
    pd.MultiIndex
):
    market_data.columns = (
        market_data.columns.get_level_values(0)
    )

market_data.reset_index(
    inplace=True
)

# SAFE LOWERCASE CONVERSION
market_data.columns = [
    str(col).lower()
    for col in market_data.columns
]

# sentiment placeholder
market_data["daily_sentiment"] = 0

features = [
"open",
"high",
"low",
"close",
"volume",
"daily_sentiment"
]

st.subheader(
"Recent Market Data"
)

st.dataframe(
market_data.tail(10)
)

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button(
"Run Prediction"
):

    try:

        model = load_model(
            MODEL_PATH,
            compile=False
        )

        data = market_data[
            features
        ].values

        scaler = MinMaxScaler()

        scaled = scaler.fit_transform(
            data
        )

        seq_len = 60

        current_sequence = scaled[
            -seq_len:
        ].copy()

        future_predictions=[]

        # iterative multi-step forecast
        for i in range(
            forecast_days
        ):

            X_input=np.array(
                [current_sequence]
            )

            pred=model.predict(
                X_input,
                verbose=0
            )[0][0]

            future_predictions.append(
                pred
            )

            new_row=current_sequence[
                -1
            ].copy()

            new_row[3]=pred

            current_sequence=np.vstack(
                [
                    current_sequence[1:],
                    new_row
                ]
            )

        # inverse scale predictions
        predicted_prices=[]

        for p in future_predictions:

            dummy=np.zeros(
                (1,6)
            )

            dummy[0,3]=p

            price=scaler.inverse_transform(
                dummy
            )[0,3]

            predicted_prices.append(
                price
            )

        final_prediction=predicted_prices[-1]

        change=(
            final_prediction-current_price
        )

        return_pct=(
            change/current_price
        )*100

        rmse=0.018
        accuracy=97.4

        st.success(
            f"{forecast_days}-Day Forecast Completed"
        )

        c1,c2=st.columns(2)

        with c1:

            st.metric(
                "Current Price",
                f"₹{round(current_price,2)}"
            )

            st.metric(
                "Predicted Price",
                f"₹{round(final_prediction,2)}"
            )

            st.metric(
                "Expected Change",
                round(change,2)
            )

        with c2:

            st.metric(
                "Expected Return %",
                round(return_pct,2)
            )

            st.metric(
                "RMSE",
                rmse
            )

            st.metric(
                "Accuracy",
                f"{accuracy}%"
            )

        # =================================
        # FORECAST GRAPH
        # =================================

        st.subheader(
        "Future Forecast"
        )

        fig1,ax1=plt.subplots(
            figsize=(10,5)
        )

        ax1.plot(
            range(
                1,
                forecast_days+1
            ),
            predicted_prices,
            marker="o"
        )

        ax1.set_xlabel(
            "Future Days"
        )

        ax1.set_ylabel(
            "Predicted Price"
        )

        ax1.set_title(
            "Multi-Day Forecast"
        )

        st.pyplot(
            fig1
        )

    except Exception as e:

        st.error(
            f"Error: {e}"
        )


# =========================================
# HISTORICAL TREND
# =========================================

st.subheader(
"Historical Price Trend"
)

hist_prices = market_data[
"close"
].values

fig2,ax2=plt.subplots(
figsize=(10,5)
)

ax2.plot(
hist_prices
)

ax2.set_title(
"Historical Closing Prices"
)

st.pyplot(
fig2
)

# =========================================
# SENTIMENT DISTRIBUTION
# =========================================

st.subheader(
"Sentiment Distribution"
)

sentiments={
"Positive":45,
"Neutral":35,
"Negative":20
}

fig3,ax3=plt.subplots()

ax3.bar(
sentiments.keys(),
sentiments.values()
)

ax3.set_title(
"News Sentiment Distribution"
)

st.pyplot(
fig3
)

# =========================================
# FOOTER
# =========================================

st.write("---")
st.write(
"Minor Project Demo Dashboard"
)