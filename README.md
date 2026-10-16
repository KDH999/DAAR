<div align="center">

# DAAR

**Leveraging Large Language Models for Domain-Agnostic Aspect-Aware Recommendation**

Official implementation of **Domain-Agnostic Aspect-Aware Recommendation (DAAR)**

[Overview](#overview) · [Architecture](#model-architecture) · [Getting Started](#getting-started) · [Results](#experimental-results) · [Citation](#citation)

</div>

---

## Overview

DAAR is a review-based rating prediction framework that uses large language models to identify **aspect terms** without predefined aspect categories. It combines their semantic representations and aspect-level sentiment scores with user–item interactions for personalized rating prediction.

<p align="center">
  <img src="assets/Figure 2.png" alt="Figure 2. Overall framework of DAAR" width="850">
</p>
<p align="center"><em>Figure 2. Overall framework of DAAR.</em></p>

## Model Architecture

The recommender system comprises three modules:

1. **User–Item Interaction:** learns user and item representations from their IDs.
2. **Sentiment-Aware Aspect Encoding:** combines Phrase-BERT aspect embeddings with DeBERTa sentiment probabilities and applies multi-head attention.
3. **Rating Prediction:** integrates interaction and aspect representations through an MLP to predict ratings.

<p align="center">
  <img src="model/Figure 3.png" alt="Figure 3. DAAR recommender system architecture" width="780">
</p>
<p align="center"><em>Figure 3. Recommender system architecture of DAAR.</em></p>

Model implementation: [`model/daar.py`](model/daar.py) · [Architecture details](model/README.md)

## Repository Structure

```text
DAAR/
├── assets/              # Framework figure and documentation
├── data/                # Data format documentation
├── model/               # DAAR architecture and implementation
├── preprocessing/       # ATE, aspect embeddings, and sentiment analysis
├── config.yaml          # Model and training configuration
├── train.py             # Model training
├── evaluate.py          # Evaluation (MAE, MSE, RMSE)
├── requirements.txt     # Python dependencies
└── README.md
```

## Getting Started

### Requirements

Install dependencies and the spaCy English model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Aspect extraction uses `meta-llama/Llama-3.1-8B-Instruct`, which requires Hugging Face model access. Set `HF_TOKEN` before running ATE. A CUDA-compatible GPU is recommended for LLaMA inference.

### Datasets

The experiments use the following datasets:

| Dataset | Source |
| --- | --- |
| Baby Products | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) |
| Office Products | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) |
| Yelp | [Yelp Open Dataset](https://www.yelp.com/dataset) |

Raw datasets are not redistributed. See [Data Preparation](data/README.md) for input fields and the Yelp subset description.

### Preprocessing

The following example uses **Amazon Baby Products** with the columns `user_id`, `asin`, `rating`, and `text`. Update file paths and column arguments for other datasets.

**Step 1 — Aspect term extraction (LLaMA 3.1)**

```bash
python preprocessing/extract_aspects.py \
  --input data/Baby_Products.jsonl \
  --output data/Baby_Products_ate.json \
  --lines --drop-duplicates --remove-punctuation
```

**Step 2 — Aspect postprocessing (75th-percentile `K_max`)**

```bash
python preprocessing/postprocess_aspects.py \
  --input data/Baby_Products_ate.json \
  --output data/Baby_Products_postprocessed.json \
  --five-core --percentile 75
```

**Step 3 — Aspect embeddings (Phrase-BERT)**

```bash
python preprocessing/embed_aspects.py \
  --input data/Baby_Products_postprocessed.json \
  --output data/Baby_Products_embeddings.json
```

**Step 4 — Aspect sentiment classification (DeBERTa)**

```bash
python preprocessing/sentiment_analysis.py \
  --input data/Baby_Products_embeddings.json \
  --output data/Baby_Products_final.json
```

For details about each stage, see [Preprocessing](preprocessing/README.md).

### Training and Evaluation

Set the final preprocessed data path and column names in [`config.yaml`](config.yaml), then run:

```bash
python train.py --config config.yaml
python evaluate.py --test-data artifacts/test_data.npz --model checkpoints/daar.keras
```

The configuration uses a **7:1:2** train/validation/test split and early stopping. The evaluation setting uses review-derived features associated with each target rating, consistent with the study.

## Experimental Results

DAAR results from **Table 2 of the manuscript** (mean ± standard deviation over five runs; lower is better):

| Dataset | MAE ↓ | RMSE ↓ |
| :--- | ---: | ---: |
| Baby Products | **0.408 ± 0.005** | **0.594 ± 0.003** |
| Office Products | **0.367 ± 0.006** | **0.540 ± 0.002** |
| Yelp | **0.521 ± 0.005** | **0.672 ± 0.003** |

The provided training script executes a single run; the table reports the five-run results in the paper.

## Citation

Publication details and a BibTeX entry will be added after publication.
