import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Movie Review Sentiment Analysis")

st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Review Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

st.write("Enter your name and movie review.")

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

# Clean Text Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.split()
    text = [word for word in text if word not in ENGLISH_STOP_WORDS]
    return " ".join(text)

# Clean Reviews
data["clean_review"] = data["review"].apply(clean_text)

# Convert Sentiment
data["sentiment"] = data["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# Features and Labels
X = data["clean_review"]
y = data["sentiment"]

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)


# User Input
name = st.text_input("Enter Your Name")

review = st.text_area(
    "Enter Movie Review",
    placeholder="Example: The movie was fantastic and acting was amazing!"
)

# Button
if st.button("Analyze Review"):

    if name == "":
        st.warning("Please enter your name.")

    elif review == "":
        st.warning("Please enter movie review.")

    else:

        cleaned_review = clean_text(review)

        vectorized_review = vectorizer.transform([cleaned_review])

        prediction = model.predict(vectorized_review)[0]

        probability = model.predict_proba(vectorized_review)[0]

        confidence = max(probability) * 100

        st.subheader("Prediction Result")

        st.write(f"Thank you {name}!")

        if prediction == 1:
            st.success("Predicted Sentiment: Positive 😊")
        else:
            st.error("Predicted Sentiment: Negative 😞")

        st.info(f"Confidence in this prediction: {confidence:.2f}%")