import pickle
from src.preprocess import clean_text

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    confidence = max(probs)

    return prediction, confidence