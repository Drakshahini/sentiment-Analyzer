import pandas as pd

# Load dataset
df = pd.read_csv("data/reviews.csv")

# Keep only required columns
df = df[['label', 'tweet']]

# Rename columns
df.columns = ['label', 'text']

# Convert labels (0 → negative, 1 → positive)
df['label'] = df['label'].map({
    0: 'negative',
    1: 'positive'
})

# Save cleaned dataset
df.to_csv("data/reviews.csv", index=False)

print("✅ Dataset cleaned successfully!")