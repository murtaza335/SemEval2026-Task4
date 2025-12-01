# src/train.py

import os
from src.dataloader import load_data
from src.preprocess import preprocess_texts
from src.model_baseline_TFIDF_LR import build_model
from src.evaluate import evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    # Determine root directory and correct data path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "dev_track_a.jsonl")

    print("Loading data from:", data_path)

    # 1. Load data
    texts, labels = load_data(data_path)

    # 2. Preprocess
    texts_clean = preprocess_texts(texts)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 4. Build model
    vectorizer, clf = build_model()

    # 5. Vectorize
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6. Train model
    clf.fit(X_train_vec, y_train)

    # 7. Evaluate
    print("\n--- Evaluation on Test Set ---")
    evaluate_model(clf, X_test_vec, y_test)

    # 8. Confusion matrix
    y_pred = clf.predict(X_test_vec)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.show()


if __name__ == "__main__":
    main()
