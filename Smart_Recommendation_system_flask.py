from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\tripadvisor_hotel_reviews.csv")
df.rename(columns={"Review": "cleaned_review", "Rating": "review_score"}, inplace=True)

def map_rating_to_sentiment(rating):
    return 2 if rating >= 4 else (1 if rating == 3 else 0)

df["sentiments"] = df["review_score"].apply(map_rating_to_sentiment)
df["cleaned_review"].fillna("No Review", inplace=True)

df["user_id"] = np.random.randint(1, 1000, df.shape[0])
df["hotel_id"] = np.random.randint(1, 100, df.shape[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
model.to(device)
model.eval()

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["cleaned_review"])

app = Flask(__name__)

def predict_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**encoding)
        pred_label = torch.argmax(output.logits, dim=1).item()
    return {0: "Negative", 1: "Neutral", 2: "Positive"}[pred_label]

@app.route("/")
def home():
    return render_template("index_recommender.html")

@app.route("/predict_sentiment", methods=["POST"])
def predict():
    data = request.json
    review_text = data.get("review", "")
    sentiment = predict_sentiment(review_text)
    return jsonify({"review": review_text, "sentiment": sentiment})

@app.route("/top_hotels", methods=["GET"])
def recommend_top_hotels():
    top_hotels = df[df["sentiments"] == 2]["hotel_id"].value_counts().index.tolist()[:10]
    return jsonify({"top_hotels": [int(hotel) for hotel in top_hotels]})

@app.route("/recommend/<int:user_id>", methods=["GET"])
def recommend_hotels(user_id):
    user_reviews = df[df["user_id"] == user_id]
    liked_hotels = user_reviews[user_reviews["sentiments"] == 2]["hotel_id"].tolist()

    if not liked_hotels:
        return recommend_top_hotels()

    similar_users = df[(df["hotel_id"].isin(liked_hotels)) & (df["sentiments"] == 2)]["user_id"].unique()
    recommendations = df[(df["user_id"].isin(similar_users)) & (df["sentiments"] == 2)]["hotel_id"].unique()
    recommendations = list(set(recommendations) - set(liked_hotels))
    
    return jsonify({"recommended_hotels": [int(hotel) for hotel in recommendations[:10]] if recommendations else recommend_top_hotels()})

@app.route("/similar_hotels/<int:hotel_id>", methods=["GET"])
def recommend_similar_hotels(hotel_id):
    try:
        hotel_idx = df[df["hotel_id"] == hotel_id].index[0]
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        similar_hotels = np.argsort(-cosine_similarities[hotel_idx])[1:6]
        return jsonify({"similar_hotels": df.iloc[similar_hotels]["hotel_id"].tolist()})
    except:
        return jsonify({"error": "Hotel ID not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
