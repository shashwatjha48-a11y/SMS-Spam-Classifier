import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Data cleaning
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)

# Convert ham/spam to 0/1
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["target"] = encoder.fit_transform(df["target"])

# Remove duplicate messages
df = df.drop_duplicates(keep="first")

# Text preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)


df["transformed_text"] = df["text"].apply(transform_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["transformed_text"]).toarray()
y = df["target"].values

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X, y)

# Save the FITTED vectorizer and model
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("===================================")
print("Model training completed!")
print("vectorizer.pkl created successfully")
print("model.pkl created successfully")
print("===================================")
