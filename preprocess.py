import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text
