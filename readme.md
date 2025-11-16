# Comprehensive EDA for SemEval Narrative Similarity

This repository contains the **Exploratory Data Analysis (EDA)** performed for the **SemEval Narrative Similarity** project.  
The goal of this EDA is to understand the structure, distribution, and characteristics of the dataset—particularly the *anchor*, *text_a*, and *text_b* fields—and to visualize patterns that may support modeling decisions later.

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

The dataset used is:
- **dev_track_a.jsonl** 
- Each record consists of:
  - `anchor_text` — The main narrative or reference story  
  - `text_a` — Narrative A to compare with anchor  
  - `text_b` — Narrative B to compare with anchor  
  - Additional metadata may be included depending on the SemEval version

# How to Run the EDA Script
Follow the steps below to run the comprehensive EDA for the SemEval Narrative Similarity dataset.

## Install Dependencies
Make sure you have Python 3.8+ installed.
Install all required libraries using:
pip install -r requirements
The requirements are all mentioned in requirements.txt

## Place the Dataset
Ensure the dataset file is located in the project directory: dev_track_a.jsonl

## Run the EDA Script
Execute the Python file that generates all exploratory data analysis visualizations: comprehensive_eda.py
This script will:
- Load the dataset  
- Process the text fields  
- Generate multiple EDA plots  
- Save all visualizations into the `plots/` directory  
- Print useful statistics and summaries  

## View the Generated Plots
All output plots (PDFs or PNGs) will appear in: /plots






