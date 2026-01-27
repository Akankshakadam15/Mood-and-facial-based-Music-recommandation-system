import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Load dataset
df = pd.read_csv("spotify_millsongdata.csv")

# Clean lyrics
def clean(text):
    text = str(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

df["cleaned"] = df["text"].apply(clean)

# Sentiment score
def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    return score

df["sentiment"] = df["cleaned"].apply(get_sentiment)

# Create emotion label
def emotion_label(score):
    if score > 0.6:
        return "Happy"
    elif score > 0.2:
        return "Romantic"
    elif score > -0.2:
        return "Calm"
    elif score > -0.6:
        return "Sad"
    else:
        return "Dark"

df["emotion"] = df["sentiment"].apply(emotion_label)

df.to_csv("emotion_labeled_dataset.csv", index=False)

print("Emotion labels generated successfully!")
print(df.head())
