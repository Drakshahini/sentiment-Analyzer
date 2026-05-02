from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    score = sia.polarity_scores(text)

    compound = score['compound']

    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound