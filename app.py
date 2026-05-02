import streamlit as st
import pandas as pd
from model import predict_sentiment
from utils import clean_text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

# ----------- PAGE CONFIG -----------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# ----------- 🌈 RAINBOW BACKGROUND + ANIMATIONS -----------
st.markdown("""
<style>

/* 🌈 Soft Animated Gradient */
.stApp {
    background: linear-gradient(
        120deg,
        #fdfbfb,
        #e0f7fa,
        #fce4ec,
        #f3e5f5,
        #e8f5e9
    );
    background-size: 300% 300%;
    animation: gradientMove 10s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
    animation: fadeIn 1s ease-in;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 20px;
    animation: fadeIn 2s ease-in;
}

/* Result Box */
.box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
    backdrop-filter: blur(10px);
    animation: fadeInUp 0.6s ease-in;
}

/* Colors */
.positive { background-color: rgba(212, 237, 218, 0.8); color: #155724; }
.negative { background-color: rgba(248, 215, 218, 0.8); color: #721c24; }
.neutral  { background-color: rgba(255, 243, 205, 0.8); color: #856404; }

/* Animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# ----------- HEADER -----------
st.markdown('<div class="title">📊 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions instantly using AI</div>', unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/2920/2920244.png", width=120)

# ----------- INPUT TYPE -----------
option = st.radio("Choose Input Type", ["Text", "CSV Upload"])

# ----------- TEXT INPUT -----------
if option == "Text":
    text = st.text_area("Enter text")

    if st.button("Analyze"):
        with st.spinner("Analyzing sentiment... ⏳"):
            time.sleep(1.5)

        cleaned = clean_text(text)
        result, confidence = predict_sentiment(cleaned)

        if result == "positive":
            st.balloons()
            st.markdown(
                f'<div class="box positive">😊 Positive (Confidence: {confidence:.2f})</div>',
                unsafe_allow_html=True
            )
        elif result == "negative":
            st.markdown(
                f'<div class="box negative">😞 Negative (Confidence: {confidence:.2f})</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="box neutral">😐 Neutral (Confidence: {confidence:.2f})</div>',
                unsafe_allow_html=True
            )

# ----------- CSV INPUT -----------
elif option == "CSV Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        with st.spinner("Processing CSV... ⏳"):
            time.sleep(1.5)

        df = pd.read_csv(file)

        if 'text' not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            df['clean_text'] = df['text'].apply(clean_text)

            results = []
            confidences = []

            for t in df['clean_text']:
                res, conf = predict_sentiment(t)
                results.append(res)
                confidences.append(conf)

            df['sentiment'] = results
            df['confidence'] = confidences

            st.subheader("📄 Results")
            st.write(df)

            # 📊 Donut Chart
            st.subheader("📊 Sentiment Distribution")
            counts = df['sentiment'].value_counts()

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

            # ☁️ Word Cloud
            st.subheader("☁️ Word Cloud")
            text_all = " ".join(df['text'].astype(str))
            wc = WordCloud(width=800, height=400).generate(text_all)
            st.image(wc.to_array())