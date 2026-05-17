# Part 3: NLP and Sequence Modeling Mini Project

## Overview

This project builds a complete NLP pipeline to classify customer support messages by sentiment (positive, neutral, negative). It compares traditional text vectorization methods with sequence-based deep learning approaches, and explains why transformers have become the dominant architecture for NLP.

## Dataset

**File:** `customer_support_text_classification.csv`  
**Source:** Part 3 dataset (Masai Module 5 Assignment)  
**Records:** 1,500 customer support tickets

| Column | Description |
|--------|-------------|
| `customer_message` | Raw text of the support message |
| `sentiment_label` | Target: `positive`, `neutral`, `negative` |
| `channel` | Source: chat, phone, email, social |
| `word_count` | Token count of message |
| `urgent_flag` | Binary urgency indicator |

**Class Distribution:** Negative: 497 (33.1%) | Neutral: 524 (34.9%) | Positive: 479 (31.9%)  
Near-perfect balance — no augmentation or resampling required.

## Approach

### Text Preprocessing
1. Lowercasing
2. Removing numbers and special characters (`re.sub(r'[^a-z\s]', '', text)`)
3. Whitespace normalization

### Vectorization Methods
| Method | Description | Matrix Shape |
|--------|-------------|-------------|
| TF-IDF | Weighted word frequency (bigrams included) | (1200, 5000) |
| Bag of Words | Raw word counts | (1200, 5000) |
| Tokenizer + Padding | Integer sequences for LSTM | (1200, 50) |

### Models Trained

| Model | Vectorizer | Weighted F1 |
|-------|-----------|-------------|
| Logistic Regression | TF-IDF (bigrams) | 1.0000 |
| Naive Bayes | Bag of Words | 1.0000 |
| **LSTM** | Embedding + Tokenizer | ~0.99+ |

> The near-perfect F1 reflects the synthetic, pattern-rich nature of this dataset. On real-world customer messages with slang, sarcasm, and mixed sentiment, sequence models significantly outperform bag-of-words approaches.

## LSTM Architecture

```
Input Sequence (max_len=50 tokens)
    → Embedding(vocab=10000, dim=128)
    → LSTM(64 units)
    → Dense(64, ReLU)
    → Dropout(0.4)
    → Dense(3, Softmax)
```

## Attention and Transformer Concepts

| Concept | Key Insight |
|---------|-------------|
| **RNN limitation** | Gradients vanish over long sequences — early context is lost |
| **LSTM advantage** | Cell state with forget/input/output gates preserves relevant long-range context |
| **Attention** | Decoder attends directly to any encoder position — no bottleneck through single vector |
| **Transformer** | Full self-attention + parallelism enables scaling to billions of parameters |

## Repository Structure

```
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.csv
    ├── model_evaluation.png
    └── sample_predictions.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Place `customer_support_text_classification.csv` in the same directory as the notebook.
