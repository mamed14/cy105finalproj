import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# App title
st.title("🎬 Movie Taste Predictor (Letterboxd-Powered)")

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
if model:
    title = st.text_input("Movie Title")
    tags = st.text_input("Tags (optional)", placeholder="e.g. sci-fi, thriller")

    if st.button("Predict"):
        input_features = vectorizer.transform([f"{title} {tags}"])
        distances, indices = model.kneighbors(input_features)
        predicted_ratings = df.iloc[indices[0]]["Rating"]
        avg_score = predicted_ratings.mean()
        similar = df.iloc[indices[0]]["Name"].tolist()

        st.markdown(f"### 🎯 Predicted Score: **{round(avg_score, 2)} / 5**")
        st.markdown("#### 🎥 Similar Movies You've Rated:")
        st.write(similar)
