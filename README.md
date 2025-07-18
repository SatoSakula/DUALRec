# DUALRec: A Hybrid Sequential and Language Model Framework for Context-Aware Movie Recommendation

This repository contains the official implementation of **DUALRec**, developed as part of my Master's thesis at the University of Amsterdam. DUALRec is a hybrid recommendation framework that combines the sequential modeling strengths of LSTM with the semantic understanding of Language Models (LLMs) to improve movie recommendations in a context-aware setting.

---

##  Key Features

- **Sequential Modeling**: Learns user behavior patterns using an LSTM-based architecture.
- **Language Model Integration**: Enhances recommendations with semantic alignment using transformer-based LLMs.
- **Hybrid Architecture**: Combines ID-based sequential prediction with genre-aware semantic re-ranking.
- **Multi-level Evaluation**: Evaluates both ranking accuracy and genre diversity.

---

##  Repository Structure

```
DUALRec/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── data_loader.py         # Load and preprocess MovieLens 1M data
│   ├── lstm_model.py          # LSTM architecture for sequence modeling
│   ├── llm_integration.py     # Cosine similarity and genre-aware reranking with LLMs
│   ├── evaluation.py          # HR@K, NDCG, Genre Jaccard similarity, etc.
│   └── train.py               # Training pipeline
└── notebook/
    └── exploratory_analysis.ipynb
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DUALRec.git
cd DUALRec
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the training script

```bash
python src/train.py
```

---

## Evaluation Metrics

We evaluate our model using both accuracy- and diversity-based metrics:
- `HR@K` (Hit Rate)
- `NDCG@K` (Normalized Discounted Cumulative Gain)
- `Genre Jaccard Similarity` (semantic alignment)

---

## Dataset

This project uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). Please download them from the grouplens website.

---

## Acknowledgements

- Pretrained LLMs used via HuggingFace `transformers` and `sentence-transformers`.
- LoRA and lightweight tuning inspired by PEFT library from HuggingFace.

---
