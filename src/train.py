import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.preprocess import clean_text

def load_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['id', 'entity', 'sentiment', 'text']

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)

    # Remove irrelevant class
    df = df[df['sentiment'] != 'irrelevant']

    df['sentiment'] = df['sentiment'].str.lower()
    df['clean_text'] = df['text'].apply(clean_text)

    return df[['clean_text', 'sentiment']]

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_df = load_data("data/twitter_training.csv")
    val_df = load_data("data/twitter_validation.csv")

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['sentiment']

    X_val = vectorizer.transform(val_df['clean_text'])
    y_val = val_df['sentiment']

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\nClassification Report:\n")
    report = classification_report(y_val, y_pred)
    print(report)

    # Save report
    with open("outputs/metrics.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Save model
    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()