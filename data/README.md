# Data

The original Yelp and Amazon review datasets are not redistributed in this repository.

Prepare a JSON file with one interaction per record and the following fields:

```text
user_id
asin
rating
aspects
embeddings
sentiments
```

- `aspects`: extracted aspect terms
- `embeddings`: corresponding aspect-term embeddings
- `sentiments`: corresponding 3-dimensional sentiment probability vectors

`K_max` is determined from the 75th percentile of the aspect-count distribution. Sequences longer than `K_max` are truncated and shorter sequences are zero-padded before being passed to the model.
