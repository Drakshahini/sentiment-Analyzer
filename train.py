import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df['text'] = df['text'].apply(clean_text)

# Balance dataset
df = df.groupby('label').apply(lambda x: x.sample(n=5000, replace=True)).reset_index(drop=True)

# Features & labels
X = df['text']
y = df['label']

# Vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression(max_iter=300, class_weight='balanced')
model.fit(X_vec, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Improved model trained!")