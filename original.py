import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# App title
st.title("🎬 Movie Taste Predictor (Letterboxd-Powered)")
st.markdown("""
    <style>
        body {
            background-color: #fffde7;
        }
        .main {
            background-color: #fffde7;
        }
    </style>
""", unsafe_allow_html=True)

# Global vars
model = None
vectorizer = None
df = pd.DataFrame()

# Upload CSV
uploaded_file = st.file_uploader("Upload your Letterboxd `ratings.csv`", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "Name" not in df.columns or "Rating" not in df.columns:
        st.error("CSV must have 'Name' and 'Rating' columns.")
    else:
        df.dropna(subset=["Name", "Rating"], inplace=True)
        df["Tags"] = df.get("Tags", "")
        df["features"] = df["Name"] + " " + df["Tags"].fillna("")

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["features"])
        model = NearestNeighbors(n_neighbors=5, metric="cosine")
        model.fit(X)
        st.success("Model trained! Enter a movie below to see if you'd like it.")

# Predict section
0 commit comments
Comments
0
 (0)
Comment
You're receiving notifications because you're subscribed to this thread.

R20 selected.  
