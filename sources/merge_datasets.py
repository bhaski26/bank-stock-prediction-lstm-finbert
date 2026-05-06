import pandas as pd

# ================================
# 1. FUNCTION TO LOAD STOCK DATA
# ================================

def load_stock_data(filepath, bank_name):

    # Fix Yahoo Finance multi-header issue
    df = pd.read_csv(filepath, skiprows=2)

    # Rename Date column
    df.rename(columns={"Date": "date"}, inplace=True)

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Add bank column
    df["bank"] = bank_name

    # Drop rows with invalid dates
    df = df.dropna(subset=["date"])

    return df


# ================================
# 2. LOAD ALL BANK STOCK DATA
# ================================

sbi = load_stock_data(
    r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/sbi_stock_data.csv",
    "State Bank of India"
)

hdfc = load_stock_data(
    r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/hdfc_stock_data.csv",
    "HDFC Bank"
)

icici = load_stock_data(
    r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/icici_stock_data.csv",
    "ICICI Bank"
)

axis = load_stock_data(
    r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/axis_stock_data.csv",
    "Axis Bank"
)

kotak = load_stock_data(
    r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/kotak_stock_data.csv",
    "Kotak Mahindra Bank"
)

# Combine all stock data
stock_df = pd.concat([sbi, hdfc, icici, axis, kotak], ignore_index=True)

print("\nStock dataset loaded")
print(stock_df.head())


# ================================
# 3. LOAD SENTIMENT DATA
# ================================

sentiment_df = pd.read_csv(r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets/news_with_sentiment.csv")

sentiment_df["date"] = pd.to_datetime(
    sentiment_df["date"],
    errors="coerce"
).dt.date

print("\nSentiment dataset loaded")
print(sentiment_df.head())


# ================================
# 4. AGGREGATE SENTIMENT
# ================================

daily_sentiment = sentiment_df.groupby(
    ["bank", "date"]
)["sentiment_score"].mean().reset_index()

daily_sentiment.rename(
    columns={"sentiment_score": "daily_sentiment"},
    inplace=True
)

print("\nDaily sentiment calculated")
print(daily_sentiment.head())


# ================================
# 5. MERGE DATASETS
# ================================

merged_df = pd.merge(
    stock_df,
    daily_sentiment,
    on=["bank", "date"],
    how="left"
)

# Fill missing sentiment with neutral
merged_df["daily_sentiment"] = merged_df["daily_sentiment"].fillna(0)

print("\nMerged dataset preview")
print(merged_df.head())


# ================================
# 6. SAVE FINAL DATASET
# ================================

merged_df.to_csv(
    "datasets/final_stock_sentiment_dataset.csv",
    index=False
)

print("\nFinal dataset created successfully!")