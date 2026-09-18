# Leveraging Large Language Models for Domain-Agnostic Aspect-Aware Recommendation

Official implementation of **DAAR (Domain-Agnostic Aspect-Aware Recommendation)**.

DAAR extracts aspect terms from review text using LLaMA 3.1, represents the extracted terms with Phrase-BERT and aspect-level sentiment probabilities, and combines them with user-item interactions for rating prediction.

## Overview

DAAR integrates LLM-based aspect term extraction, sentiment-aware aspect representation, multi-head attention, and user-item interaction modeling for rating prediction.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

LLaMA 3.1-8B-Instruct requires Hugging Face access. Set the access token as an environment variable:

```bash
export HF_TOKEN=YOUR_TOKEN
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="YOUR_TOKEN"
```

## Repository Structure

```text
DAAR/
├── data/
│   └── README.md
├── model/
│   └── daar.py
├── preprocessing/
│   ├── extract_aspects.py
│   ├── postprocess_aspects.py
│   ├── embed_aspects.py
│   ├── sentiment_analysis.py
│   └── prepare_model_input.py
├── config.yaml
├── train.py
├── evaluate.py
├── requirements.txt
└── .gitignore
```

## Data Preparation

The experiments use the Yelp and Amazon review datasets. Raw datasets are not included in this repository.

Each review record should contain a user ID, item ID, rating, and review text. See `data/README.md` for the expected format.

## Preprocessing

### 1. Aspect Term Extraction

```bash
python preprocessing/extract_aspects.py \
  --input data/Baby_Products.jsonl \
  --output data/Baby_Products_ate.json \
  --lines \
  --drop-duplicates \
  --remove-punctuation
```

The default checkpoint is:

```text
meta-llama/Llama-3.1-8B-Instruct
```

Duplicate review records are removed before text preprocessing and aspect term extraction. ATE is then performed with 4-bit NF4 quantization, `do_sample=False`, and `max_new_tokens=100`.

### 2. Aspect Postprocessing

```bash
python preprocessing/postprocess_aspects.py \
  --input data/Baby_Products_ate.json \
  --output data/Baby_Products_postprocessed.json \
  --user-column user_id \
  --item-column asin \
  --rating-column rating \
  --five-core \
  --percentile 75
```

Duplicate reviews have already been removed before ATE. Postprocessing removes extracted terms that do not occur in the source review and removes single-word terms that are not identified as nouns. Reviews with no remaining aspect terms are discarded. The dataset is then filtered using the 5-core criterion.

The maximum aspect sequence length is determined by the 75th percentile of the aspect-term count distribution. Reviews exceeding this value are truncated before aspect representation is generated.

### 3. Phrase-BERT Embedding

```bash
python preprocessing/embed_aspects.py \
  --input data/Baby_Products_postprocessed.json \
  --output data/Baby_Products_embeddings.json
```

The default checkpoint is:

```text
whaleloops/phrase-bert
```

### 4. Aspect-level Sentiment Analysis

```bash
python preprocessing/sentiment_analysis.py \
  --input data/Baby_Products_embeddings.json \
  --output data/Baby_Products_final.json
```

The default checkpoint is:

```text
yangheng/deberta-v3-base-absa-v1.1
```

For each aspect term, the three-class sentiment probabilities are used as the sentiment representation.

## Model

DAAR uses user and item IDs together with Phrase-BERT aspect embeddings and aspect-level sentiment probabilities. The sentiment probabilities are transformed into 768-dimensional vectors and combined with the corresponding aspect embeddings through element-wise multiplication. Multi-head attention is then applied to the sentiment-aware aspect representations, with an attention mask excluding zero-padded aspect positions. The resulting representation is concatenated with the user-item interaction representation for final rating prediction.

Default model settings are provided in `config.yaml`.

## Training

Set the final preprocessed dataset path in `config.yaml` and run:

```bash
python train.py --config config.yaml
```

## Evaluation

```bash
python evaluate.py \
  --test-data artifacts/test_data.npz \
  --model checkpoints/daar.keras
```

The evaluation script reports MAE, MSE, RMSE, and MAPE.

## Citation

Citation information will be added after publication.
