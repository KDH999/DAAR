# DAAR Model

`daar.py` implements the **DAAR rating-prediction network** described in Section 3.4 of the paper.

![DAAR recommender system architecture schematic](architecture.svg)

| Module | Implementation |
| --- | --- |
| User–item interaction | User and item ID embeddings, concatenation, and an MLP |
| Sentiment-aware aspect encoding | Phrase-BERT aspect embeddings multiplied elementwise with projected 3-class sentiment probabilities; multi-head attention with a padding mask |
| Rating prediction | Concatenation of the interaction and aspect representations, followed by an MLP and a linear rating output |

The model accepts encoded user IDs, item IDs, padded aspect embeddings, sentiment probabilities, and an aspect mask. `K_max` is determined by the 75th-percentile rule in `preprocessing/postprocess_aspects.py`; the model receives the resulting length from `train.py`.

The illustration above is a schematic rendering of the paper's recommender architecture. See [`../README.md`](../README.md) for setup, training, and evaluation.
