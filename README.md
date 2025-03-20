# Smart Hotel Recommendation System

## Overview
This project is a **Smart Hotel Recommendation System** that uses **sentiment analysis** and **recommendation algorithms** to provide personalized hotel suggestions. It utilizes **TripAdvisor hotel reviews** as the dataset and implements both **user-based collaborative filtering** and **content-based filtering** to suggest hotels. The system is built with **Flask** for the backend and integrates a **BERT-based sentiment analysis model** to classify user reviews.

## Dataset
- **Source**: TripAdvisor Hotel Reviews
- **Columns**:
  - `Review` (text of the hotel review) → Processed as `cleaned_review`
  - `Rating` (user rating from 1-5) → Mapped to `review_score`
  - `sentiments` (derived from ratings):
    - 0: Negative (rating ≤ 2)
    - 1: Neutral (rating = 3)
    - 2: Positive (rating ≥ 4)
  - `user_id` (randomly assigned unique users)
  - `hotel_id` (randomly assigned hotel IDs)

## Justification for Recommendation Approach
### 1. User-Based Collaborative Filtering (Using User ID)
- The system tracks **users who liked the same hotels**.
- If a user has previously liked certain hotels, the system finds **similar users** and recommends hotels they enjoyed.
- If a user has no history, **default top-rated hotels** are recommended.

### 2. Content-Based Filtering (Using Hotel ID)
- Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** on reviews to find similarity.
- When a user searches for a hotel, the system recommends **similar hotels** based on review content.

## API Endpoints
### 1. Home Page
`GET /`
- Loads the **index_recommender.html** template.

### 2. Sentiment Analysis
`POST /predict_sentiment`
- **Input:** JSON `{ "review": "Amazing hotel with great service!" }`
- **Output:** `{ "review": "Amazing hotel with great service!", "sentiment": "Positive" }`

### 3. Top Hotels
`GET /top_hotels`
- Returns **top 10 highest-rated hotels**.

### 4. Personalized Hotel Recommendation
`GET /recommend/<int:user_id>`
- **Finds hotels liked by similar users**.
- If a user has **no previous history**, returns top-rated hotels.

### 5. Similar Hotel Recommendation
`GET /similar_hotels/<int:hotel_id>`
- Returns hotels **similar to the given hotel** based on **content-based filtering**.

## Model Performance
- **Sentiment Classification Model**
  - **Accuracy:** 94%
  - Model: **Fine-tuned BERT**
  - Predicts sentiment as **Positive, Neutral, or Negative**

## Notes
- **Initial Load Time**: Since the system loads a **BERT model** for sentiment analysis, **please wait a few minutes** when starting the application.
- **Data Processing**: User IDs and hotel IDs are **randomly assigned**, meaning the recommendations will differ upon reloading the dataset.

## Running the Application
1. Install dependencies:
   ```bash
   pip install flask torch transformers pandas numpy scikit-learn
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

