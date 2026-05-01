import io
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.predict import predict_sentiment

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #1f4037, #99f2c8);
}
.main {
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 15px;
}
h1, h2, h3 {
    color: #2c3e50;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
}
.footer {
    text-align: center;
    color: gray;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/twitter_training.csv", header=None)
df.columns = ['id', 'entity', 'sentiment', 'text']

df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
df['sentiment'] = df['sentiment'].str.strip().str.lower()
df = df[df['sentiment'] != 'irrelevant']

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔍 Filters")

entities = ["All"] + sorted(df['entity'].unique())
selected = st.sidebar.selectbox("Select Entity", entities)

if selected != "All":
    df = df[df['entity'] == selected]

search = st.sidebar.text_input("Search keyword")

if search:
    df = df[df['text'].str.contains(search, case=False, na=False)]

# ---------------- HEADER ----------------
st.markdown("## 📊 Social Media Sentiment Analysis Dashboard")
st.caption("Real-time insights using Machine Learning")

# ---------------- KPIs ----------------
counts = df['sentiment'].value_counts()
total = len(df)

positive = counts.get('positive', 0)
negative = counts.get('negative', 0)
positive_pct = (positive / total) * 100 if total > 0 else 0
negative_pct = (negative / total) * 100 if total > 0 else 0

col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><h3>Total Posts</h3><h2>{total}</h2></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h3>Positive %</h3><h2>{positive_pct:.1f}%</h2></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h3>Negative %</h3><h2>{negative_pct:.1f}%</h2></div>", unsafe_allow_html=True)

# ---------------- CHARTS ----------------
st.markdown("### 📊 Sentiment Distribution")

fig_bar = px.bar(
    counts,
    x=counts.index,
    y=counts.values,
    color=counts.index,
    color_discrete_sequence=["#ff4b4b", "#f9c74f", "#4CAF50"]
)

st.plotly_chart(fig_bar, use_container_width=True)

fig_pie = px.pie(
    values=counts.values,
    names=counts.index,
    color_discrete_sequence=["#ff4b4b", "#f9c74f", "#4CAF50"]
)

st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- SAMPLE TWEETS ----------------
st.markdown("### 📝 Sample Tweets")

if total > 0:
    st.dataframe(df[['text', 'sentiment']].sample(min(10, total)))
else:
    st.info("No tweets match the selected filters.")

# ---------------- TOP KEYWORDS ----------------
st.markdown("### 🔥 Top Keywords")

top_pos = ['love','best','fun','excited','nice']
top_neg = ['worst','sucks','fix','boring']
top_neu = ['latest','check','earned']

c1, c2, c3 = st.columns(3)

c1.markdown("🟢 **Positive**")
c1.markdown(" ".join([f"`{w}`" for w in top_pos]))

c2.markdown("⚪ **Neutral**")
c2.markdown(" ".join([f"`{w}`" for w in top_neu]))

c3.markdown("🔴 **Negative**")
c3.markdown(" ".join([f"`{w}`" for w in top_neg]))

# ---------------- LIVE PREDICTION ----------------
st.markdown("### 🔍 Try Live Prediction")

user_input = st.text_area("💬 Enter text to analyze", height=100)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result, confidence = predict_sentiment(user_input)
    
    if result == "positive":
        st.success(f"😊 Positive ({confidence:.2f})")
    elif result == "negative":
        st.error(f"😡 Negative ({confidence:.2f})")
    else:
        st.info(f"😐 Neutral ({confidence:.2f})")

# ---------------- INSIGHTS ----------------
st.markdown("### 📌 Key Insights")

st.markdown("""
- 🔴 Negative sentiment slightly dominates → potential issues  
- 🟢 Positive sentiment strong → good engagement  
- ⚪ Neutral → ongoing discussions  
""")

# ---------------- DOWNLOAD REPORT ----------------
st.markdown("### 📥 Download Report")

report_buffer = io.StringIO()
report_buffer.write(f"""
Total Posts: {total}
Positive %: {positive_pct:.2f}
Negative %: {negative_pct:.2f}

Insights:
- Negative sentiment slightly dominates
- Positive engagement is strong
""")
report = report_buffer.getvalue()

st.download_button(
    label="Download Report",
    data=report,
    file_name="sentiment_report.txt",
    mime="text/plain"
)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div class='footer'>Built by Varda 🚀 | ML + NLP Project</div>", unsafe_allow_html=True)
