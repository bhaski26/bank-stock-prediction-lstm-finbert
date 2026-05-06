from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime
import re

# Replace with your NewsAPI key
API_KEY = "25e6073749d04a828650a0e17de28855"

newsapi = NewsApiClient(api_key=API_KEY)

# Banks used in your project
banks = [
    "State Bank of India",
    "HDFC Bank",
    "ICICI Bank",
    "Axis Bank",
    "Kotak Mahindra Bank"
]

all_news = []

for bank in banks:

    articles = newsapi.get_everything(
        q=bank,
        language="en",
        sort_by="publishedAt",
        page_size=100
    )

    for article in articles["articles"]:
        news_data = {
            "bank": bank,
            "date": article["publishedAt"],
            "headline": article["title"],
            "description": article["description"],
            "content": article["content"],
            "source": article["source"]["name"],
            "url": article["url"]
        }

        all_news.append(news_data)

# Convert to DataFrame
news_df = pd.DataFrame(all_news)

# Convert date format
news_df["date"] = pd.to_datetime(news_df["date"]).dt.date


# ================================
# NEWS DATA CLEANING SECTION
# ================================

# Combine headline + description + content
news_df["text"] = (
    news_df["headline"].fillna("") + " " +
    news_df["description"].fillna("") + " " +
    news_df["content"].fillna("")
)

# Function to clean text
def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning
news_df["clean_text"] = news_df["text"].apply(clean_text)

# Remove rows where cleaned text is empty
news_df = news_df[news_df["clean_text"] != ""]

# Remove duplicates
news_df = news_df.drop_duplicates(subset="clean_text")


# ================================
# SAVE CLEAN DATASET
# ================================

news_df.to_csv("clean_bank_news_dataset.csv", index=False)

print("News data collected and cleaned successfully")
print(news_df.head())