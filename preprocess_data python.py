
# -----------------------------
# INSTALL KAGGLE (WINDOWS ONLY)
# -----------------------------
import subprocess
subprocess.run(["pip", "install", "kaggle"], shell=True)

# -----------------------------
# IMPORTING DEPENDENCIES
# -----------------------------
import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# -----------------------------
# KAGGLE CONFIG (Windows)
# -----------------------------
# Make kaggle folder
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

# Copy kaggle.json to correct path
import shutil
shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))

# Set permissions
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Load config
kaggle_config = json.load(open("kaggle.json"))
os.environ["KAGGLE_USERNAME"] = kaggle_config["username"]
os.environ["KAGGLE_KEY"] = kaggle_config["key"]

# -----------------------------
# DOWNLOAD DATASET (NO ! , USE subprocess)
# -----------------------------
print("\nDownloading dataset from Kaggle...")

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "notshrirang/spotify-million-song-dataset"
])

# -----------------------------
# EXTRACT ZIP
# -----------------------------
print("\nExtracting ZIP file...")

with zipfile.ZipFile('spotify-million-song-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("spotify_millsongdata.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

df["text"] = df["text"].fillna("")

# -----------------------------
# TOP ARTISTS
# -----------------------------
top_artists = df["artist"].value_counts().head(10)
print("\nTop 10 Artists:")
print(top_artists)

# SAMPLE DATA FOR SPEED
df = df.sample(10000).reset_index(drop=True)

# Drop link column if exists
if "link" in df.columns:
    df = df.drop("link", axis=1)

# -----------------------------
# WORD CLOUD
# -----------------------------
all_lyrics = " ".join(df["text"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_lyrics)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Lyrics")
plt.show()

# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(preprocess_text)

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["cleaned_text"])

# COSINE SIMILARITY
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_songs(song_name, cosine_sim=cosine_sim, df=df, top_n=5):
    idx = df[df["song"].str.lower() == song_name.lower()].index
    if len(idx) == 0:
        return "Song not found in the dataset!"
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    song_indices = [i[0] for i in sim_scores]

    return df[["artist", "song"]].iloc[song_indices]

# Test recommendation
print("\nRecommendations:")
print(recommend_songs("For The First Time"))

# -----------------------------
# ADD LYRICS LENGTH COLUMN
# -----------------------------
df["lyrics_length"] = df["text"].apply(lambda x: len(str(x).split()))

# -----------------------------
# PLOTS (Matplotlib)
# -----------------------------
plt.figure(figsize=(10, 5))
plt.hist(df["lyrics_length"], bins=50, edgecolor="black")
plt.title("Lyrics Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Songs")
plt.show()

# BAR PLOT
top_artists = df["artist"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_artists.index, y=top_artists.values)
plt.title("Top 10 Artists")
plt.xlabel("Artist Name")
plt.ylabel("Number of Songs")
plt.xticks(rotation=45)
plt.show()

# PIE CHART
top5 = df["artist"].value_counts().head(5)
plt.figure(figsize=(7, 7))
plt.pie(top5.values, labels=top5.index, autopct="%1.1f%%")
plt.title("Top 5 Artists Distribution")
plt.show()

# BOX PLOT
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["lyrics_length"])
plt.title("Lyrics Length Spread")
plt.xlabel("Number of Words")
plt.show()

# HEATMAP
plt.figure(figsize=(6, 4))
sns.heatmap(df[["lyrics_length"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# SCATTER PLOT
plt.figure(figsize=(10, 5))
plt.scatter(range(len(df)), df["lyrics_length"], s=10, alpha=0.5)
plt.title("Scatter Plot of Lyrics Length")
plt.xlabel("Song Index")
plt.ylabel("Lyrics Length")
plt.show()

# WORD FREQUENCY BAR CHART
all_words = " ".join(df["text"]).lower()
all_words = re.findall(r"\b[a-z]+\b", all_words)
word_freq = Counter(all_words).most_common(20)

words = [w for w, f in word_freq]
freq = [f for w, f in word_freq]

plt.figure(figsize=(12, 5))
sns.barplot(x=words, y=freq)
plt.title("Top 20 Most Frequent Words in Lyrics")
plt.xticks(rotation=45)
plt.show()
