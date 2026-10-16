# Model Architecture

The DAAR recommender system is implemented in [`daar.py`](daar.py).

<p align="center">
  <img src="Figure 3.png" alt="DAAR recommender system architecture" width="780">
</p>
<p align="center"><em>Figure 3. Recommender system architecture of DAAR.</em></p>

| Module | Description |
| --- | --- |
| **User–Item Interaction** | User and item ID embeddings are concatenated and passed through an MLP. |
| **Sentiment-Aware Aspect Encoding** | Phrase-BERT aspect vectors are multiplied by projected three-class sentiment probabilities, then processed with masked multi-head attention. |
| **Rating Prediction** | Interaction and aspect representations are concatenated and passed through an MLP to predict the rating. |

Inputs comprise encoded user/item IDs, padded aspect embeddings, sentiment probabilities, and an aspect mask. The maximum sequence length, `K_max`, is determined during aspect postprocessing using the 75th-percentile rule.

[← Back to the main README](../README.md)
