# Data

The original Yelp and Amazon review datasets are not redistributed in this repository.

Before preprocessing, each interaction should contain:

```text
user identifier
item identifier
rating
review text
```

Column names can differ by dataset. For example, the Amazon experiments use `user_id`, `asin`, `rating`, and `text`; Yelp uses its corresponding user/item/rating columns.

After preprocessing, the final model input also contains:

```text
aspects
embeddings
sentiments
```

- `aspects`: aspect terms retained after ATE postprocessing and 75th-percentile truncation
- `embeddings`: corresponding 768-dimensional Phrase-BERT embeddings
- `sentiments`: corresponding three-dimensional DeBERTa sentiment-probability vectors

`K_max` is determined during preprocessing from the 75th percentile of the aspect-count distribution after the filtering steps. Aspect lists longer than `K_max` are truncated before representation extraction. During model-input construction, shorter sequences are zero-padded to the retained maximum length.

See the repository root `README.md` for the complete preprocessing and training commands.
