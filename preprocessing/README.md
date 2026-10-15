# Preprocessing

The DAAR preprocessing stages are applied in the following order:

| Script | Output |
| --- | --- |
| `extract_aspects.py` | Aspect terms extracted using LLaMA 3.1 |
| `postprocess_aspects.py` | Filtered aspects and dataset-specific `K_max` truncation |
| `embed_aspects.py` | 768-dimensional Phrase-BERT vectors |
| `sentiment_analysis.py` | Three-class DeBERTa sentiment probabilities |
| `prepare_model_input.py` | Padded model inputs, mask, and encoded user/item IDs (used by `train.py`) |

The complete commands and options are documented in the [root README](../README.md). The extraction and postprocessing scripts use dataset-specific column arguments; for example, Yelp uses `business_id` and `stars` instead of the Amazon example's `asin` and `rating`.
