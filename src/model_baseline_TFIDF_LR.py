# src/model_baseline_TFIDF_LR.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
    vectorizer = TfidfVectorizer(
        max_features=50000,     # bigger vocabulary
        ngram_range=(1, 2),     # bigrams help classification a lot
        min_df=2,               # ignore extremely rare words
        sublinear_tf=True       # good for LR performance
    )

    clf = LogisticRegression(
        C=2.0,                  # slightly weaker regularization
        max_iter=2000,          # more stable convergence
        solver="liblinear",     # best for small-medium datasets
        class_weight=None       # (adjust to "balanced" if dataset is imbalanced)
    )

    return vectorizer, clf
