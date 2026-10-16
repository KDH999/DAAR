<div align="center">

# DAAR

**Leveraging Large Language Models for Domain-Agnostic Aspect-Aware Recommendation**

Official implementation of **Domain-Agnostic Aspect-Aware Recommendation (DAAR)**.

</div>

## Overview

DAAR uses LLaMA 3.1 to extract aspect terms from reviews, Phrase-BERT to encode them, and DeBERTa to estimate aspect-level sentiment. A sentiment-aware attention network combines these representations with user–item interactions for rating prediction.

<p align="center">
  <img src="assets/Figure 2.png" alt="Figure 2. Overall framework of DAAR" width="780">
</p>
<p align="center"><em>Figure 2. Overall framework of DAAR.</em></p>

The recommender architecture (Figure 3) and its implementation are available in the [model directory](model/README.md).

## Repository Structure

| Directory | Description |
| --- | --- |
| [preprocessing/](preprocessing/README.md) | Aspect extraction, embedding, and sentiment analysis |
| [model/](model/README.md) | DAAR architecture and implementation |
| [data/](data/README.md) | Dataset sources and input formats |

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Prepare a dataset following [Data Preparation](data/README.md) and run the four stages in [Preprocessing](preprocessing/README.md). LLaMA 3.1-8B-Instruct requires Hugging Face model access (`HF_TOKEN`).

Set the preprocessed dataset path in `config.yaml`, then train and evaluate:

```bash
python train.py --config config.yaml
python evaluate.py --test-data artifacts/test_data.npz --model checkpoints/daar.keras
```

The evaluation uses review-derived features associated with the target rating, as in the paper.

## Experimental Results

DAAR performance (mean ± standard deviation over five runs; lower is better):

| Dataset | MAE ↓ | RMSE ↓ |
| --- | ---: | ---: |
| Baby Products | **0.408 ± 0.005** | **0.594 ± 0.003** |
| Office Products | **0.367 ± 0.006** | **0.540 ± 0.002** |
| Yelp | **0.521 ± 0.005** | **0.672 ± 0.003** |

## Citation

Citation information will be added after publication.
