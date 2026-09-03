# DAAR

Official implementation of **DAAR: Domain-Agnostic Aspect-Aware Recommendation**.

This repository contains the TensorFlow implementation of the DAAR recommendation model and the preprocessing required to construct fixed-length aspect inputs from extracted aspect representations.

## Repository structure

```text
DAAR/
├── data/
│   └── README.md
├── model/
│   └── daar.py
├── preprocessing/
│   └── prepare_model_input.py
├── config.yaml
├── train.py
├── evaluate.py
├── requirements.txt
└── .gitignore
```

## Input format

The model expects a JSON file containing the following fields for each interaction:

- `user_id`: user identifier
- `asin`: item identifier
- `rating`: observed rating
- `aspects`: extracted aspect terms
- `embeddings`: aspect-term embeddings
- `sentiments`: aspect-level sentiment probability vectors

The aspect sequence length is set using the 75th percentile of the number of aspect terms in the dataset. Reviews with more aspects are truncated, while shorter reviews are zero-padded. A boolean mask is generated for valid aspect positions.

## Training

Update the dataset path in `config.yaml`, then run:

```bash
python train.py --config config.yaml
```

## Evaluation

```bash
python evaluate.py --config config.yaml
```

Additional preprocessing modules for aspect extraction, phrase embedding, and aspect-level sentiment analysis will be included separately.
