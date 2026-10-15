# Data

The Amazon and Yelp raw datasets are **not included** in this repository.

| Dataset | User ID | Item ID | Rating | Review text |
| --- | --- | --- | --- | --- |
| Amazon (Baby, Office) example | `user_id` | `asin` | `rating` | `text` |
| Yelp review records | `user_id` | `business_id` | `stars` | `text` |

Column names depend on the downloaded dataset version. Rename columns or pass the corresponding arguments to the ATE and postprocessing scripts. For Yelp, select the Indiana restaurant-review subset before preprocessing.

After preprocessing, each record also includes:

- `aspects`: retained aspect terms, truncated using the 75th percentile of aspect counts (`K_max`)
- `embeddings`: corresponding 768-dimensional Phrase-BERT vectors
- `sentiments`: corresponding three-dimensional sentiment probability vectors

During training, sequences shorter than `K_max` are zero-padded and an aspect mask distinguishes valid positions.

See the [root README](../README.md) for the complete pipeline and example commands.
