# Preprocessing

DAAR converts reviews into aspect-level representations in four stages.

| Stage | Script | Output |
| --- | --- | --- |
| 1. Aspect Term Extraction | [`extract_aspects.py`](extract_aspects.py) | Aspect terms extracted by LLaMA 3.1 |
| 2. Postprocessing | [`postprocess_aspects.py`](postprocess_aspects.py) | Filtered aspect terms truncated to `K_max` |
| 3. Aspect Embedding | [`embed_aspects.py`](embed_aspects.py) | 768-dimensional Phrase-BERT embeddings |
| 4. Sentiment Classification | [`sentiment_analysis.py`](sentiment_analysis.py) | Negative, neutral, and positive probabilities |

[`prepare_model_input.py`](prepare_model_input.py) converts the processed records into fixed-length tensors, with zero-padding and an attention mask, for training and evaluation.

The scripts accept dataset-specific column names where needed. Full example commands are provided in the [main README](../README.md).

[← Back to the main README](../README.md)
