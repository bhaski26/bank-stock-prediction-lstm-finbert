import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ============================
# 1. LOAD NEWS DATASET
# ============================

news_df = pd.read_csv(r"C:\Users\bhask\Downloads\Minor Project 2025-26\Datasets\clean_bank_news_dataset.csv")

print("News dataset loaded")
print(news_df.head())


# ============================
# 2. LOAD FINBERT MODEL
# ============================

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

model.eval()


# ============================
# 3. SENTIMENT FUNCTION
# ============================

def get_sentiment(text):

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits).item()

    # Convert to sentiment score
    if prediction == 2:
        return 1   # Positive
    elif prediction == 1:
        return 0   # Neutral
    else:
        return -1  # Negative


# ============================
# 4. RUN SENTIMENT ANALYSIS
# ============================

sentiments = []

for text in tqdm(news_df["clean_text"]):
    sentiments.append(get_sentiment(text))

news_df["sentiment_score"] = sentiments


# ============================
# 5. SAVE RESULT DATASET
# ============================

news_df.to_csv("news_with_sentiment.csv", index=False)

print("Sentiment analysis completed")
print(news_df.head())