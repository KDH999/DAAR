# Data Preparation

The datasets used in the paper are publicly available and are **not redistributed** in this repository.

| Dataset | Source |
| --- | --- |
| Baby Products, Office Products | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) |
| Yelp | [Yelp Open Dataset](https://www.yelp.com/dataset) |

## Required Input Fields

Each record needs a user identifier, item identifier, rating, and review text. Column names may vary by dataset version.

| Dataset | User | Item | Rating | Review |
| --- | --- | --- | --- | --- |
| Amazon example | `user_id` | `asin` | `rating` | `text` |
| Yelp reviews | `user_id` | `business_id` | `stars` | `text` |

For Yelp, prepare the Indiana restaurant-review subset before running the preprocessing scripts. Map dataset-specific column names through the CLI options and `config.yaml`.

## Generated Fields

| Field | Description |
| --- | --- |
| `aspects` | Extracted aspect terms, filtered and truncated to `K_max` |
| `embeddings` | Corresponding 768-dimensional Phrase-BERT vectors |
| `sentiments` | Corresponding three-class sentiment probabilities |

`K_max` is set using the 75th percentile of aspect counts after filtering. Shorter sequences are padded with zero vectors and masked when constructing model inputs.

[← Back to the main README](../README.md)
