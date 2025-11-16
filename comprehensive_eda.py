import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings("ignore")

# --- Transformers & Models ---
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import umap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# --- CREATE PLOTS FOLDER ---
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# 1. LOAD DATA
print("Loading dataset...")
with open("dev_track_a.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

print(f"Loaded {len(df)} records.")
print("\nColumns:", df.columns.tolist())
print("\nSample:")
print(df.head(2))

# 2. BASIC EDA
print("\n" + "="*50)
print("BASIC EDA")
print("="*50)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
label_counts = df['text_a_is_closer'].value_counts()
print(label_counts)

plt.figure(figsize=(6, 4))
sns.countplot(x='text_a_is_closer', data=df, palette="viridis")
plt.title('Distribution of Target Labels (text_a_is_closer)')
plt.xlabel('Is Text A Closer?')
plt.ylabel('Count')
plt.xticks([0, 1], ['Text B Closer', 'Text A Closer'])
plt.savefig(f"{plots_dir}/label_distribution.pdf")
plt.show()

# 3. TOKEN LENGTH ANALYSIS
print("\n" + "="*50)
print("TOKEN LENGTH ANALYSIS")
print("="*50)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

token_data = []
for record in tqdm(data, desc="Tokenizing texts"):
    anchor_len = len(tokenizer.encode(record["anchor_text"], truncation=False))
    a_len = len(tokenizer.encode(record["text_a"], truncation=False))
    b_len = len(tokenizer.encode(record["text_b"], truncation=False))
    token_data.append({"anchor_tokens": anchor_len, "text_a_tokens": a_len, "text_b_tokens": b_len})

token_df = pd.DataFrame(token_data)

print("\nAverage token lengths:")
print(token_df.mean().round(2))
print("\nMax token lengths:")
print(token_df.max())

# Melt for plotting
token_melted = token_df.melt(var_name="Text Type", value_name="Token Count")

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Text Type", y="Token Count", data=token_melted, palette="viridis")
plt.title("Token Length Distribution (Anchor vs Text A vs Text B)")
plt.ylabel("Token Count")
plt.savefig(f"{plots_dir}/token_length_boxplot.pdf")
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=token_melted, x="Token Count", hue="Text Type", kde=True, element="step", palette="viridis")
plt.title("Token Length Histogram")
plt.xlabel("Token Count")
plt.savefig(f"{plots_dir}/token_length_histogram.pdf")
plt.show()

# 4. TF-IDF + COSINE SIMILARITY + LDA
print("\n" + "="*50)
print("TF-IDF & LDA ANALYSIS")
print("="*50)

texts = df["anchor_text"].tolist() + df["text_a"].tolist() + df["text_b"].tolist()
n = len(df)

# TF-IDF
tfidf_vect = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vect.fit_transform(texts)

anchor_vecs = tfidf_matrix[:n]
a_vecs = tfidf_matrix[n:2*n]
b_vecs = tfidf_matrix[2*n:]

sim_a_tfidf = [cosine_similarity(anchor_vecs[i], a_vecs[i])[0][0] for i in range(n)]
sim_b_tfidf = [cosine_similarity(anchor_vecs[i], b_vecs[i])[0][0] for i in range(n)]

# LDA
count_vect = CountVectorizer(stop_words='english', max_features=5000)
X_count = count_vect.fit_transform(texts)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
topics = lda.fit_transform(X_count)

anchor_topics = topics[:n]
a_topics = topics[n:2*n]
b_topics = topics[2*n:]

lda_sim_a = [cosine_similarity(anchor_topics[i].reshape(1, -1), a_topics[i].reshape(1, -1))[0][0] for i in range(n)]
lda_sim_b = [cosine_similarity(anchor_topics[i].reshape(1, -1), b_topics[i].reshape(1, -1))[0][0] for i in range(n)]

# TF-IDF Scatter
plt.figure(figsize=(7, 7))
plt.scatter(sim_a_tfidf, sim_b_tfidf, alpha=0.6, color='teal')
plt.plot([0, 0.35], [0, 0.35], '--', color='gray', label='y = x')
plt.xlim(0, 0.35)
plt.ylim(0, 0.35)
plt.xlabel("Anchor vs Text A (TF-IDF Cosine)")
plt.ylabel("Anchor vs Text B (TF-IDF Cosine)")
plt.title("TF-IDF Similarity Scatter Plot")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{plots_dir}/tfidf_similarity_scatter.pdf")
plt.show()

# 5. JACCARD & BAG-OF-WORDS COSINE
print("\n" + "="*50)
print("VOCABULARY OVERLAP: JACCARD + COSINE")
print("="*50)

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', text.lower()).strip()

def jaccard_sim(t1, t2):
    s1, s2 = set(clean_text(t1).split()), set(clean_text(t2).split())
    return len(s1 & s2) / len(s1 | s2) if s1 or s2 else 0

def bow_cosine(t1, t2):
    vec = CountVectorizer().fit([t1, t2])
    v1, v2 = vec.transform([t1, t2])
    return cosine_similarity(v1, v2)[0][0]

df['jaccard_a'] = df.apply(lambda r: jaccard_sim(r['anchor_text'], r['text_a']), axis=1)
df['jaccard_b'] = df.apply(lambda r: jaccard_sim(r['anchor_text'], r['text_b']), axis=1)
df['cosine_a_bow'] = df.apply(lambda r: bow_cosine(r['anchor_text'], r['text_a']), axis=1)
df['cosine_b_bow'] = df.apply(lambda r: bow_cosine(r['anchor_text'], r['text_b']), axis=1)

# Scatter: Jaccard vs BoW Cosine
plt.figure(figsize=(10, 5))
plt.scatter(df['jaccard_a'], df['cosine_a_bow'], color='blue', alpha=0.6, label='Text A', s=50)
plt.scatter(df['jaccard_b'], df['cosine_b_bow'], color='red', alpha=0.6, label='Text B', s=50)
plt.xlabel('Jaccard Similarity')
plt.ylabel('Bag-of-Words Cosine Similarity')
plt.title('Vocabulary Overlap: Anchor vs A/B')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"{plots_dir}/vocab_overlap_scatter.pdf")
plt.show()

# 6. SEMANTIC SIMILARITY (Sentence-BERT)
print("\n" + "="*50)
print("SEMANTIC SIMILARITY (all-MiniLM-L6-v2)")
print("="*50)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

results = []
for record in tqdm(data, desc="Computing semantic similarity"):
    anchor_emb = model.encode(record["anchor_text"], convert_to_tensor=True)
    a_emb = model.encode(record["text_a"], convert_to_tensor=True)
    b_emb = model.encode(record["text_b"], convert_to_tensor=True)

    sim_a = util.cos_sim(anchor_emb, a_emb).item()
    sim_b = util.cos_sim(anchor_emb, b_emb).item()

    predicted = "text_a" if sim_a > sim_b else "text_b"
    true = "text_a" if record["text_a_is_closer"] else "text_b"

    results.append({
        "sim_a": sim_a, "sim_b": sim_b,
        "predicted": predicted, "true": true
    })

sem_df = pd.DataFrame(results)
accuracy = (sem_df['predicted'] == sem_df['true']).mean()
print(f"\nSemantic Model Accuracy: {accuracy*100:.2f}%")

# Distribution of predictions
plt.figure(figsize=(8, 5))
sns.countplot(x=sem_df['predicted'], palette="viridis")
plt.title("Predicted Closer Text (Semantic Similarity)")
plt.xlabel("Predicted Closer")
plt.savefig(f"{plots_dir}/semantic_prediction_distribution.pdf")
plt.show()

# Similarity per record
plt.figure(figsize=(12, 6))
x = range(len(sem_df))
plt.scatter(x, sem_df['sim_a'], label="Sim Anchor → A", alpha=0.7)
plt.scatter(x, sem_df['sim_b'], label="Sim Anchor → B", alpha=0.7)
plt.title("Semantic Similarity per Record")
plt.xlabel("Record Index")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.savefig(f"{plots_dir}/semantic_similarity_per_record.pdf")
plt.show()

# 7. CLUSTER VISUALIZATION (UMAP)
print("\n" + "="*50)
print("CLUSTERING EMBEDDINGS WITH UMAP")
print("="*50)

all_texts = df['anchor_text'].tolist() + df['text_a'].tolist() + df['text_b'].tolist()
all_emb = model.encode(all_texts, show_progress_bar=True, batch_size=32)

labels_type = (['anchor'] * n) + (['text_a'] * n) + (['text_b'] * n)
labels_true = np.repeat(df['text_a_is_closer'].values, 3)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
proj = reducer.fit_transform(all_emb)

vis_df = pd.DataFrame({
    'x': proj[:, 0],
    'y': proj[:, 1],
    'type': labels_type,
    'true_label': labels_true.astype(str)
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=vis_df,
    x='x', y='y',
    hue='type',
    style='true_label',
    palette='Set2',
    s=70,
    alpha=0.8
)
plt.title("UMAP Projection of Sentence Embeddings (Anchor, A, B)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Type | True: A closer?")
plt.savefig(f"{plots_dir}/umap_projection.pdf")
plt.show()

# 8. FINAL SUMMARY
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Total Records: {len(df)}")
print(f"Label Balance: {label_counts.to_dict()}")
print(f"Semantic Model Accuracy: {accuracy*100:.2f}%")
print("\nAll plots saved in the folder 'plots'.")
