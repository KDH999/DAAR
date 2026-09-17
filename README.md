# DAAR

Official implementation of **DAAR: Domain-Agnostic Aspect-Aware Recommendation**.

DAAR extracts domain-agnostic aspect terms from review text with an instruction-tuned LLM, represents the extracted phrases with Phrase-BERT, incorporates aspect-level sentiment probabilities, and predicts user ratings with a multi-head attention based recommendation model.

## Pipeline

```text
Review text
   │
   ├─ 1. LLM-based Aspect Term Extraction
   │
   ├─ 2. Aspect postprocessing + 5-core filtering
   │      └─ determine K_max from the 75th percentile
   │         and truncate aspect lists before representation learning
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

The 75th-percentile truncation is performed during preprocessing. The recommendation model does **not** select or truncate aspect terms again. It only zero-pads shorter aspect sequences to the maximum length already present in the preprocessed data so that samples can be batched into fixed-size tensors.

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

The code was implemented in Python with TensorFlow for recommendation and PyTorch/Transformers for text preprocessing.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The LLaMA checkpoint requires Hugging Face access. Set your token as an environment variable instead of writing it in source code:

```bash
export HF_TOKEN=YOUR_TOKEN
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="YOUR_TOKEN"
```

## Data

The experiments use review data containing at least the following fields before preprocessing:

- `user_id`: user identifier
- `asin`: item identifier
- `rating`: observed rating
- `text`: review text

The final model input JSON additionally contains:

- `aspects`: extracted and postprocessed aspect terms
- `embeddings`: Phrase-BERT vectors for the retained aspect terms
- `sentiments`: three-dimensional sentiment probability vectors for each aspect

Raw datasets are not redistributed in this repository. See `data/README.md` for dataset information.

## Preprocessing

### 1. Aspect term extraction

```bash
python preprocessing/extract_aspects.py \
  --input data/Baby_Products.jsonl \
  --output data/Baby_Products_ate.json \
  --lines
```

The extraction script uses an instruction-tuned LLaMA model in 4-bit quantization with deterministic decoding (`do_sample=False`).

### 2. Postprocessing and 75th-percentile truncation

```bash
python preprocessing/postprocess_aspects.py \
  --input data/Baby_Products_ate.json \
  --output data/Baby_Products_postprocessed.json \
  --five-core \
  --drop-duplicates \
  --percentile 75
```

Postprocessing follows the experimental notebook:

1. Remove extracted terms that do not occur in the source review.
2. For single-word terms, retain noun terms based on spaCy POS tagging.
3. Remove reviews with no remaining aspect terms.
4. Remove duplicate interactions and apply 5-core filtering.
5. Compute the 75th percentile of the number of aspect terms per review.
6. Truncate reviews exceeding this value while retaining all shorter reviews.

For the Baby experiment, the resulting maximum retained aspect length was **7**.

### 3. Phrase-BERT embeddings

```bash
python preprocessing/embed_aspects.py \
  --input data/Baby_Products_postprocessed.json \
  --output data/Baby_Products_embeddings.json
```

The default checkpoint is `whaleloops/phrase-bert`, producing 768-dimensional aspect-term representations.

### 4. Aspect-level sentiment probabilities

```bash
python preprocessing/sentiment_analysis.py \
  --input data/Baby_Products_embeddings.json \
  --output data/Baby_Products_final.json
```

The default sentiment checkpoint is `yangheng/deberta-v3-base-absa-v1.1`. For each aspect-review pair, the three-class softmax probabilities are retained and used directly by DAAR rather than reducing them to a discrete polarity label.

## Model

DAAR takes four inputs:

1. encoded user ID
2. encoded item ID
3. Phrase-BERT aspect representations
4. aspect-level sentiment probability vectors

The sentiment probability vector is transformed through fully connected layers and multiplied element-wise with the corresponding aspect representation. Multi-head self-attention then aggregates the sentiment-aware aspect representations. The resulting review representation is concatenated with the user-item interaction representation for rating prediction.

Padding is used only to construct fixed-size batch tensors. No attention mask is applied to padded positions in the released implementation, matching the original experimental code.

## Training

Set the final preprocessed dataset path in `config.yaml` and run:

```bash
python train.py --config config.yaml
```

Default Baby experiment settings in `config.yaml` include:

- train/test split: 80/20 with `random_state=42`
- validation split: 0.125 of the training portion, yielding an overall 7:1:2 split
- user/item embedding dimension: 128
- aspect embedding dimension: 768
- attention heads: 6
- attention key dimension: 128
- learning rate: 1e-4
- batch size: 64
- maximum epochs: 100
- early-stopping patience: 5

The trained model is saved under `checkpoints/`, and the exact held-out test tensors are saved under `artifacts/`.

## Evaluation

```bash
python evaluate.py \
  --test-data artifacts/test_data.npz \
  --model checkpoints/daar.keras
```

The script reports MAE, MSE, RMSE, and MAPE.

## Notes on reproducibility

The scripts in this repository are organized versions of the experimental notebooks used to develop DAAR. Dataset paths, access tokens, intermediate files, and machine-specific paths have been removed from the public code. Preprocessing and model hyperparameters are exposed through command-line arguments or `config.yaml` where appropriate.

## Citation

Citation information will be added after publication.
