import pandas as pd
import joblib
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------------------
# LOAD MODEL + TF-IDF + DATASET
# ----------------------------------------

print("Loading model...")
model = joblib.load("emotion_model.pkl")
tfidf = joblib.load("tfidf.pkl")

print("Loading dataset...")
df = pd.read_csv("emotion_labeled_dataset.csv")

# Drop NaN
df = df.dropna(subset=["cleaned", "emotion"])

X = df["cleaned"]
y_true = df["emotion"]


# ----------------------------------------
# PREDICT EMOTIONS
# ----------------------------------------

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text)).lower()
    return text

print("Transforming dataset...")
X_tfidf = tfidf.transform(X)

print("Predicting...")
y_pred = model.predict(X_tfidf)


# ----------------------------------------
# ACCURACY
# ----------------------------------------

accuracy = accuracy_score(y_true, y_pred)
print("\n============ MODEL ACCURACY ============")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ----------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------

print("\n============ CLASSIFICATION REPORT ============")
print(classification_report(y_true, y_pred))

# ----------------------------------------
# CONFUSION MATRIX
# ----------------------------------------

print("\n============ CONFUSION MATRIX ============")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Emotion Classification Confusion Matrix")
plt.show()


# ----------------------------------------
# SAVE PREDICTIONS FOR REVIEW
# ----------------------------------------

results = df.copy()
results["predicted_emotion"] = y_pred
results.to_csv("evaluation_results.csv", index=False)

print("\nPredictions saved to evaluation_results.csv")
print("Evaluation complete!")
