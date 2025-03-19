import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r"C:\Users\sarthak mohapatra\Downloads\archive\tripadvisor_hotel_reviews.csv")
df.rename(columns={"Review": "cleaned_review", "Rating": "review_score"}, inplace=True)

def map_rating_to_sentiment(rating):
    if rating >= 4:
        return 2
    elif rating == 3:
        return 1
    else:
        return 0

df["sentiments"] = df["review_score"].apply(map_rating_to_sentiment)
df["cleaned_review"].fillna("No Review", inplace=True)

class_weights = compute_class_weight("balanced", classes=np.unique(df["sentiments"]), y=df["sentiments"])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class HotelDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        encoding = tokenizer(self.reviews[idx], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_texts, test_texts, train_labels, test_labels = train_test_split(df["cleaned_review"].tolist(), df["sentiments"].tolist(), test_size=0.2)

train_dataset = HotelDataset(train_texts, train_labels)
test_dataset = HotelDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# optimizer = optim.AdamW(model.parameters(), lr=2e-5)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# for epoch in range(5):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
    
#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         optimizer.zero_grad()
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = criterion(outputs.logits, labels)
        
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = torch.argmax(outputs.logits, dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     scheduler.step()
#     train_acc = correct / total
#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Train Accuracy = {train_acc:.4f}")

# torch.save(model.state_dict(), "bert_sentiment_model.pth")

# def predict_sentiment(model, text):
#     model.eval()
#     with torch.no_grad():
#         encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
#         output = model(**encoding)
#         pred_label = torch.argmax(output.logits, dim=1).item()
#         sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
#         return sentiment_map[pred_label]

model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
model.to(device)

# test_reviews = [
#     "The hotel was fantastic, great service and clean rooms!",
#     "Decent place but the food could be better.",
#     "Terrible experience, will never stay here again.",
#     "Nice location, comfortable stay.",
#     "The rooms were okay, but the staff was rude.",
#     "The hotel was not bad."
# ]

# for review in test_reviews:
#     sentiment = predict_sentiment(model, review)
#     print(f"Review: {review} -> Sentiment: {sentiment}")

df["user_id"] = np.random.randint(1, 1000, df.shape[0])
df["hotel_id"] = np.random.randint(1, 100, df.shape[0])

def recommend_hotels(user_id, df):
    user_reviews = df[df["user_id"] == user_id]
    liked_hotels = user_reviews[user_reviews["sentiments"] == 2]["hotel_id"].tolist()

    if user_reviews.empty:
        print(f"User {user_id} has no reviews. Suggesting top-rated hotels.")
        return df[df["sentiments"] == 2]["hotel_id"].value_counts().index.tolist()[:10]
    # disliked_hotels = user_reviews[user_reviews["sentiments"] == 0]["hotel_id"].tolist()
    similar_users=df[(df["hotel_id"].isin(liked_hotels)) & (df["sentiments"]==2) ]["user_id"].unique()
    recommendations = df[(df["user_id"].isin(similar_users)) & (df["sentiments"]==2)]["hotel_id"].unique()
    recommendations = list(set(recommendations) - set(liked_hotels))
    if not recommendations:
        print("Recommending good hotels:")
        neutral_hotels = df[df["sentiments"] == 1]["hotel_id"].unique().tolist()
        return neutral_hotels[:10]
    return list(recommendations)

user_id = 8
recommended_hotels = recommend_hotels(user_id, df)
print(f"Recommended hotels for User {user_id}: {recommended_hotels}")
