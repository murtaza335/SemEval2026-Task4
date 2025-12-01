import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fix missing tokenizer models
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)  # remove numbers too
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words])


def preprocess_texts(texts):
    return [lemmatize_text(clean_text(t)) for t in texts]
