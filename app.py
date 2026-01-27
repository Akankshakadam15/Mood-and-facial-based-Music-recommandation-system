import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepface import DeepFace
from PIL import Image
import io

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()

# ================================
# USER LOGIN / SIGNUP SYSTEM
# ================================
USER_FILE = "users.csv"
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        f.write("username,password\n")


def load_users():
    return pd.read_csv(USER_FILE)


def save_user(username, password):
    df = load_users()
    if username in df['username'].values:
        return False
    df.loc[len(df)] = [username, password]
    df.to_csv(USER_FILE, index=False)
    return True


def login_page():
    st.title("🎧 Music Recommender Login")

    menu = ["Login", "Sign Up"]
    choice = st.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login to Your Account")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            df = load_users()
            if ((df['username'] == username) & (df['password'] == password)).any():
                st.success("Login Successful! 🎉")
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Incorrect username or password")

    else:
        st.subheader("Create New Account")

        new_user = st.text_input("Choose Username")
        new_pass = st.text_input("Choose Password", type="password")

        if st.button("Sign Up"):
            if save_user(new_user, new_pass):
                st.success("Account created successfully! You can now login.")
            else:
                st.error("Username already taken!")


# ================================
# DATA LOADING & PREPROCESSING
# ================================
@st.cache_resource
def load_data():
    df = pd.read_csv("spotify_millsongdata.csv")
    # sample if large
    if len(df) > 5000:
        df = df.sample(5000, random_state=42).reset_index(drop=True)
    return df


df = load_data()


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    return text.lower()


# ensure cleaned column exists
if "cleaned_text" not in df.columns:
    df["cleaned_text"] = df["text"].apply(preprocess_text)
else:
    df["cleaned_text"] = df["cleaned_text"].fillna(df["text"].apply(preprocess_text))

# sentiment by TextBlob (keeps your previous pipeline)
def get_textblob_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0.0


df["sentiment_score"] = df["cleaned_text"].apply(get_textblob_sentiment)


def classify_emotion(score):
    if score > 0.5:
        return "Happy / Energetic"
    elif score > 0.1:
        return "Romantic / Positive"
    elif score > -0.1:
        return "Neutral / Calm"
    elif score > -0.5:
        return "Sad"
    else:
        return "Deep Sad"


df["emotion"] = df["sentiment_score"].apply(classify_emotion)


# TF-IDF (keeps existing similarity in case needed)
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["cleaned_text"].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix)


# ================================
# EMOTION DETECTION: TEXT & FACE
# ================================
def detect_user_mood_from_text(text):
    """Return (score, label_short, recommend_type, emotion_label) for text input"""
    score = analyzer.polarity_scores(text)["compound"]
    # thresholds similar to your previous function
    if score > 0.6:
        return score, "Happy", "Energetic", "Happy / Energetic"
    if score > 0.2:
        return score, "Positive", "Romantic", "Romantic / Positive"
    if score > -0.2:
        return score, "Neutral", "Calm", "Neutral / Calm"
    if score > -0.6:
        return score, "Sad", "Calm", "Sad"
    return score, "Very Sad", "Motivational", "Deep Sad"


def map_face_emotion_to_label(face_emotion):
    """Map DeepFace emotion (happy, sad, angry, neutral, surprise, fear) to project labels.
       Also return a representative sentiment score for ranking (range -1..1)."""
    e = str(face_emotion).lower()
    if "happy" in e or "joy" in e:
        return 0.8, "Happy", "Energetic", "Happy / Energetic"
    if "sad" in e or "disgust" in e:  # treat disgust/negative similar to sad
        return -0.5, "Sad", "Calm", "Sad"
    if "angry" in e:
        return -0.6, "Angry", "Calm", "Sad"  # map angry to sad/calm recommendations (you can change)
    if "surprise" in e:
        return 0.2, "Surprised", "Energetic", "Happy / Energetic"
    if "fear" in e:
        return -0.7, "Fearful", "Calm", "Deep Sad"
    if "neutral" in e:
        return 0.0, "Neutral", "Calm", "Neutral / Calm"
    # default
    return 0.0, "Neutral", "Calm", "Neutral / Calm"


def detect_face_emotion_from_image(pil_image):
    """Use DeepFace to analyze PIL image and return dominant emotion string."""
    try:
        # convert PIL image to numpy array (RGB)
        img = np.array(pil_image.convert("RGB"))
        # DeepFace expects BGR by default when working with cv2, but analyze can accept RGB with enforce_detection=False
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        # result might be dict with 'dominant_emotion' or a list
        if isinstance(result, list):
            dominant = result[0].get("dominant_emotion", None)
        else:
            dominant = result.get("dominant_emotion", None)
        return dominant
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return None


# ================================
# RECOMMENDATION (matches previous approach)
# ================================
def recommend_by_emotion_label(emotion_label, input_score, user_text, recommend_type, user_feeling, top_n=10):
    matching_songs = df[df["emotion"] == emotion_label].copy()
    if matching_songs.empty:
        return None, f"No songs found for emotion: {emotion_label}"

    # compute closeness and sort
    matching_songs["sentiment_diff"] = abs(matching_songs["sentiment_score"] - input_score)
    matching_songs = matching_songs.sort_values("sentiment_diff").head(top_n)

    results = []
    for _, row in matching_songs.iterrows():
        results.append({
            "song": row["song"],
            "artist": row["artist"],
            "emotion": row["emotion"],
            "explanation": f"""Because you said:  
*\"{user_text}\"*  
We detected your mood as: *{user_feeling}*

We recommend songs that feel: *{recommend_type}*

Song emotion: *{row['emotion']}*  
Sentiment closeness: {1 - abs(row['sentiment_score'] - input_score):.2f}
"""
        })
    return results, None


# high-level wrapper: choose between text or face
def get_recommendations(user_text: str, camera_image, prefer_camera=False, top_n=10):
    """
    If camera_image provided and prefer_camera True -> use face detection.
    Else if camera_image provided and not prefer_camera -> we still prefer text if text non-empty.
    If no text provided and camera provided -> use face.
    """
    # if user entered text and it's not empty -> use text mood detection (unless prefer_camera True)
    if user_text and user_text.strip() != "" and not prefer_camera:
        input_score, user_feeling, recommend_type, emotion_label = detect_user_mood_from_text(user_text)
        return recommend_by_emotion_label(emotion_label, input_score, user_text, recommend_type, user_feeling, top_n=top_n)
    # else try camera
    if camera_image is not None:
        # camera_image is a UploadedFile from st.camera_input -> convert to PIL
        try:
            pil_image = Image.open(io.BytesIO(camera_image.getvalue()))
        except Exception:
            # in some contexts camera_image may already be bytes-like; try direct open
            try:
                pil_image = Image.open(camera_image)
            except Exception as e:
                st.error("Couldn't read image from camera input.")
                return None, "Image read error"

        dominant = detect_face_emotion_from_image(pil_image)
        if dominant is None:
            return None, "Could not detect face emotion from the image."

        input_score, user_feeling, recommend_type, emotion_label = map_face_emotion_to_label(dominant)
        # create descriptive user_text for explanation
        user_text_desc = f"Face detected emotion: {dominant}"
        return recommend_by_emotion_label(emotion_label, input_score, user_text_desc, recommend_type, user_feeling, top_n=top_n)

    # if neither provided
    return None, "Please type how you feel or allow camera input."


# ================================
# STREAMLIT INTERFACE (UI)
# ================================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    st.sidebar.write(f"👤 Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    st.title("🎵 Mood & Face-Based Music Recommender with Explainable AI")

    st.markdown("Tell me how you feel with text, or use the camera to let me read your facial expression. Both are available — I will prefer text if you type something, otherwise I use the face photo.")

    # layout: two columns
    col1, col2 = st.columns(2)

    with col1:
        user_input = st.text_area("💬 Describe your mood or how your day was:", height=120)

        # allow user to prefer camera even if they typed something
        prefer_camera = st.checkbox("Prefer camera emotion over typed mood?", value=False)

    with col2:
        st.markdown("📷 Capture face (optional)")
        camera_image = st.camera_input("Use your webcam to capture your face (press Capture)", key="cam1")
        if camera_image is not None:
            st.image(camera_image, caption="Captured image", use_column_width=True)

    # Recommend button
    if st.button("Recommend Songs"):
        with st.spinner("Analyzing mood and finding songs..."):
            recs, error = get_recommendations(user_input, camera_image, prefer_camera=prefer_camera, top_n=10)

        if error:
            st.error(error)
        else:
            st.success("Here are songs based on detected mood:")

            for r in recs:
                st.write("---")
                st.subheader(f"🎵 {r['song']} — {r['artist']}")
                st.write(f"Emotion Category: {r['emotion']}")
                st.markdown(r["explanation"])

                # Play on Spotify link
                query = f"{r['song']} {r['artist']}"
                safe_query = urllib.parse.quote_plus(query)
                spotify_url = f"https://open.spotify.com/search/{safe_query}"
                st.markdown(f"[▶ *Play on Spotify*]({spotify_url})")

    st.markdown("---")
    st.caption("Tip: If the camera doesn't detect your face, try good lighting and hold still for a second.")