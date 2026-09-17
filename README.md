# DAAR

Official implementation of **DAAR: Domain-Agnostic Aspect-Aware Recommendation**.

DAAR extracts aspect terms from review text with an instruction-tuned LLM, represents the retained phrases with Phrase-BERT, incorporates aspect-level sentiment probabilities, and predicts user ratings with a multi-head-attention-based recommendation model.

## Pipeline

```text
Review text
   │
   ├─ 1. LLM-based Aspect Term Extraction
   │
   ├─ 2. Aspect postprocessing
   │      ├─ remove invalid aspect terms
   │      ├─ remove empty reviews
   │      ├─ 5-core filtering
   │      ├─ duplicate removal
   │      ├─ 5-core filtering again
   │      └─ determine K_max from the 75th percentile
   │         and truncate aspect lists
   │
   ├─ 3. Phrase-BERT aspect embeddings
   │
   ├─ 4. DeBERTa aspect-level sentiment probabilities
   │
   └─ 5. DAAR rating prediction
          ├─ user/item ID embeddings
          ├─ sentiment-aware aspect representations
          ├─ multi-head self-attention
          └─ MLP rating prediction
```

The 75th-percentile truncation is completed during preprocessing, before Phrase-BERT embedding and DeBERTa sentiment inference. The recommendation model does **not** select or truncate aspect terms again. It only zero-pads shorter sequences to the maximum length present in the already-truncated input data so that samples can be batched into fixed-size tensors.

## Repository structure

```text
DAAR/
├── data/
│   └── README.md
├── model/
│   ├── __init__.py
│   └── daar.py
├── preprocessing/
│   ├── __init__.py
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

## Environment

The recommendation model is implemented in TensorFlow. LLM-based ATE and aspect-level sentiment inference use PyTorch and Hugging Face Transformers.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The LLaMA checkpoint requires Hugging Face access. Set the access token as an environment variable instead of writing it in source code:

```bash
export HF_TOKEN=YOUR_TOKEN
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="YOUR_TOKEN"
```

## Data

The experiments use review data containing user IDs, item IDs, ratings, and review text. The exact item/rating column names differ between datasets (for example, Amazon uses `asin`).

The final model input additionally contains:

- `aspects`: retained aspect terms after postprocessing and 75th-percentile truncation
- `embeddings`: corresponding 768-dimensional Phrase-BERT vectors
- `sentiments`: corresponding three-dimensional sentiment probability vectors

Raw datasets are not redistributed in this repository. See `data/README.md` for dataset information.

## Preprocessing

### 1. Aspect term extraction

The supplied ATE experiment notebook loads:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

with 4-bit NF4 quantization, double quantization, FP16 computation, and deterministic generation (`do_sample=False`, `max_new_tokens=100`). The extraction prompt in `preprocessing/extract_aspects.py` follows the prompt used in the notebook.

Example:

```bash
python preprocessing/extract_aspects.py \
  --input data/Baby_Products.jsonl \
  --output data/Baby_Products_ate.json \
  --lines \
  --remove-punctuation
```

`--remove-punctuation` reproduces the explicit punctuation-removal step in the supplied ATE notebook. If additional dataset normalization was performed before the notebook input file was created, it should be reproduced during dataset preparation rather than inferred by this script.

### 2. Postprocessing and 75th-percentile truncation

Example for Amazon Baby:

```bash
python preprocessing/postprocess_aspects.py \
  --input data/Baby_Products_ate.json \
  --output data/Baby_Products_postprocessed.json \
  --user-column user_id \
  --item-column asin \
  --rating-column rating \
  --five-core \
  --drop-duplicates \
  --percentile 75
```

The supplied Baby postprocessing notebook performs the following operations:

1. Remove extracted terms that do not occur in the source review.
2. For single-word terms, retain terms identified as nouns by spaCy (`en_core_web_sm`).
3. Remove reviews with no remaining aspect terms.
4. Apply user/item 5-core filtering.
5. Remove duplicate review records.
6. Reapply 5-core filtering after duplicate removal.
7. Determine `K_max` from the 75th percentile of the retained aspect-count distribution.
8. Truncate reviews exceeding `K_max` while retaining all shorter reviews.

For the supplied Baby preprocessing notebook, the retained maximum aspect length is **7**.

### 3. Phrase-BERT embeddings

```bash
python preprocessing/embed_aspects.py \
  --input data/Baby_Products_postprocessed.json \
  --output data/Baby_Products_embeddings.json
```

The default checkpoint is:

```text
whaleloops/phrase-bert
```

which produces 768-dimensional aspect-term representations.

### 4. Aspect-level sentiment probabilities

```bash
python preprocessing/sentiment_analysis.py \
  --input data/Baby_Products_embeddings.json \
  --output data/Baby_Products_final.json
```

The sentiment checkpoint is:

```text
yangheng/deberta-v3-base-absa-v1.1
```

For each aspect term, the notebook constructs `aspect [SEP] review`, applies softmax to the three output logits, rounds the probabilities to four decimal places, and retains the probability vector as the sentiment representation.

## Model

DAAR takes four effective inputs:

1. encoded user ID
2. encoded item ID
3. Phrase-BERT aspect representations
4. aspect-level sentiment probability vectors

The three-dimensional sentiment vector is transformed through `Dense(64)` and `Dense(768)` layers and multiplied element-wise with the corresponding aspect embedding. Multi-head self-attention is then applied to the sentiment-aware aspect representations. The flattened attention output is transformed and concatenated with the user-item interaction representation for final rating prediction.

The released implementation uses:

- user/item embedding dimension: 128
- sentiment hidden dimension: 64
- multi-head attention heads: 6
- attention key dimension: 128
- aspect representation dimension: 768
- aspect feature layer: 1024
- final prediction MLP: 128 → 32 → 1

### Padding and mask

Zero-padding is used only to create fixed-size batch tensors. The supplied TensorFlow notebook constructs an `aspect_mask` array, but that mask is **not passed to `MultiHeadAttention`**. Therefore, the released model omits the unused mask input rather than implying that padded positions were masked during attention.

## Training

Set the final preprocessed dataset path and column names in `config.yaml`, then run:

```bash
python train.py --config config.yaml
```

The default Baby settings are:

- train/test split: 80/20 with `random_state=42`
- validation split: 0.125 of the training portion (overall 7:1:2)
- user/item embedding dimension: 128
- learning rate: 1e-4
- batch size: 64
- maximum epochs: 100
- early-stopping patience: 5

`train.py` performs **one training run**. The manuscript reports results averaged over five runs, so reproduction of the reported mean requires executing the experiment five times under the same experimental protocol and averaging the resulting metrics.

The trained model is saved under `checkpoints/`, and the held-out test tensors are saved under `artifacts/`.

## Evaluation

```bash
python evaluate.py \
  --test-data artifacts/test_data.npz \
  --model checkpoints/daar.keras
```

The script reports MAE, MSE, RMSE, and MAPE. The manuscript uses MAE and RMSE as the principal evaluation metrics.

## Reproducibility notes

The scripts in this repository are organized versions of the supplied experimental notebooks. Machine-specific absolute paths and access tokens have been removed. Refactoring is limited to separating notebook stages into reusable scripts, exposing paths/column names as arguments, and removing an unused attention-mask input from the model interface.

Where a manuscript statement and the supplied executable notebook differ, the repository follows the supplied notebook implementation. Such differences should be resolved in the manuscript before publication.

## Citation

Citation information will be added after publication.
