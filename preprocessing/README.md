# Preprocessing

DAAR converts review text into aspect-level representations in four stages. The commands below use Amazon Baby Products as an example (columns: `user_id`, `asin`, `rating`, `text`). Adjust dataset paths and column arguments as needed.

## 1. Aspect Term Extraction

Extract aspect terms using LLaMA 3.1-8B-Instruct (4-bit NF4, greedy decoding, `max_new_tokens=100`):

```bash
python preprocessing/extract_aspects.py --input data/Baby_Products.jsonl --output data/Baby_Products_ate.json --lines --drop-duplicates --remove-punctuation
```

`--remove-punctuation` also normalizes Unicode punctuation, symbols, and whitespace. The model requires Hugging Face access (`HF_TOKEN`).

## 2. Aspect Postprocessing

Remove invalid terms and reviews with no valid aspects, apply 5-core filtering, and truncate to the 75th-percentile aspect count (`K_max`):

```bash
python preprocessing/postprocess_aspects.py --input data/Baby_Products_ate.json --output data/Baby_Products_postprocessed.json --five-core --percentile 75
```

## 3. Aspect Embedding

Generate 768-dimensional aspect representations with Phrase-BERT:

```bash
python preprocessing/embed_aspects.py --input data/Baby_Products_postprocessed.json --output data/Baby_Products_embeddings.json
```

## 4. Aspect Sentiment Classification

Obtain three-class sentiment probabilities with DeBERTa:

```bash
python preprocessing/sentiment_analysis.py --input data/Baby_Products_embeddings.json --output data/Baby_Products_final.json
```

The training script uses [`prepare_model_input.py`](prepare_model_input.py) to encode user/item IDs, zero-pad shorter aspect sequences, and construct attention masks.

For dataset-specific columns and sources, see [Data Preparation](../data/README.md). Continue with training and evaluation in the [main README](../README.md).
