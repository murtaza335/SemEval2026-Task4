# Comprehensive EDA for SemEval Narrative Similarity

This repository contains the **Exploratory Data Analysis (EDA)** performed for the **SemEval Narrative Similarity** project.
The goal of this EDA is to understand the structure, distribution, and characteristics of the dataset—particularly the *anchor*, *text_a*, and *text_b* fields—and to visualize patterns that may support modeling decisions later.

# Team Members

* **Muhammad Murtaza (503477)** — [mmurtaza.bscs24seecs@seecs.edu.pk](mailto:mmurtaza.bscs24seecs@seecs.edu.pk)
* **Ayan Ur Rehman (518151)** — [arehman.bscs24seecs@seecs.edu.pk](mailto:arehman.bscs24seecs@seecs.edu.pk)
* **Mahnoor Khokhar (500833)** — [mkhokhar.bscs24seecs@seecs.edu.pk](mailto:mkhokhar.bscs24seecs@seecs.edu.pk)
* **Hammad Asim Kayani (513776)** — [hkayani.bscs24seecs@seecs.edu.pk](mailto:hkayani.bscs24seecs@seecs.edu.pk)

**Department of Computer Science, NUST SEECS**

# Repository Structure

```text
.
├── comprehensive_eda.py        # Main EDA script containing all analysis & plots
├── dev_track_a.jsonl           # SemEval Narrative Similarity dataset
├── requirements.txt            # Python dependencies
├── plots/                      # Folder containing exported visualizations
│   ├── plot_1.pdf
│   ├── plot_2.pdf
│   └── ...
└── README.md                   # Project documentation
```

# Dataset Description

The dataset used is **dev_track_a.jsonl**, which is part of the SemEval Narrative Similarity task.

Each record contains the following fields:

* `anchor_text` — The main narrative or reference story
* `text_a` — Candidate narrative A to compare with anchor
* `text_b` — Candidate narrative B to compare with anchor
* `text_a_is_closer` — Label indicating whether text_a is closer to anchor (boolean)
* Additional metadata may be included depending on the SemEval version

**Example record structure**:

```json
{
  "anchor_text": "Once upon a time, a king ruled a small kingdom.",
  "text_a": "The king was loved by all his subjects for his kindness.",
  "text_b": "A queen ruled a distant land with great wisdom.",
  "text_a_is_closer": true
}
```

> **Note:** The dataset file `dev_track_a.jsonl` should be placed in the project root folder.

# Features Extracted in the EDA

The comprehensive EDA script performs the following analyses:

1. **Token Length Analysis**

   * Token counts for anchor, text_a, and text_b
   * Boxplots and histograms

2. **TF-IDF Cosine Similarity**

   * Cosine similarity between anchor and candidate texts

3. **Latent Dirichlet Allocation (LDA)**

   * Topic modeling and comparison between anchor and candidates

4. **Vocabulary Overlap**

   * Jaccard similarity
   * Bag-of-Words cosine similarity

5. **Semantic Similarity**

   * Using Sentence-BERT (`all-MiniLM-L6-v2`) embeddings
   * Prediction of closer text and accuracy calculation

6. **UMAP Clustering**

   * Dimensionality reduction and 2D visualization of embeddings

# How to Run the EDA Script

## 1. Install Dependencies

Make sure Python 3.8+ is installed.
Install required libraries using:

```bash
pip install -r requirements.txt
```

## 2. Place the Dataset

Ensure the dataset file is located in the project directory:

```
dev_track_a.jsonl
```

## 3. Run the EDA Script

Execute the Python file:

```bash
python comprehensive_eda.py
```

The script will:

* Load the dataset
* Process the text fields
* Generate multiple EDA plots
* Save all visualizations into the `plots/` directory
* Print statistics and summaries

# View the Generated Plots

All plots (PDF format) will appear in:

```
plots/
```

# Outputs

The EDA produces visualizations and summaries such as:

* Label distribution (`text_a_is_closer`)
* Token length distribution for anchor, text_a, text_b
* TF-IDF similarity scatter plots
* Jaccard & Bag-of-Words cosine similarity plots
* Semantic similarity accuracy and distributions
* UMAP projection of sentence embeddings

# Acknowledgements

* Dataset provided by the **SemEval Narrative Similarity Task**
* Pretrained models from **Hugging Face Transformers** and **Sentence-Transformers**
* Visualization libraries: **Matplotlib**, **Seaborn**
* Dimensionality reduction: **UMAP**

```
```
