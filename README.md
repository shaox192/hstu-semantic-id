# Enriching Generative Recommendation with Semantic IDs

## Overview

This repository contains experiments extending the [HSTU](https://proceedings.mlr.press/v235/zhai24a.html) recommendation model with Semantic ID (SID) components to improve recommendation performance. The majority of this work is based on the excellent open-source HSTU implementation by Meta: [generative-recommenders](https://github.com/meta-recsys/generative-recommenders).

HSTU represents a breakthrough as one of the first sequential generative recommendation models, achieving remarkable success in production systems. However, like many traditional recommendation models, it takes as input arbitrary item IDs in user history sequences. This raises the question of whether HSTU can further benefit from instilling semantic meaning into item IDs -- an idea recently gaining traction under the term Semantic IDs (e.g. Google's [TIGER](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf), Kuaishou's [OneRec](https://arxiv.org/abs/2502.18965)).

**For the complete analysis and discussion of our findings, see our detailed blog post [here]().** #TODO

We conducted experiments on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (1M and 20M variants) to investigate this question. Notably, the original MovieLens dataset contains limited metadata (only title, year, and genre), making semantic embedding extraction challenging. We here extended the dataset with rich movie information including plot summaries, rich keywords, audience reviews etc from various sources. This enhanced metadata enables current LLMs to generate meaningful semantic embeddings for Semantic ID creation. These information can be embedded with modern LLMs and then quantized to produce SIDs. 

**This extended dataset is available for download [here](https://drive.google.com/file/d/1vUx7aZ7dwAhQeDq6OZ6EbmmqJgnLJfut/view?usp=drive_link), and we also provide scripts to recreate it end-to-end.**




***What’s in this repo***
- **🔧 SID pipeline:** residual quantized k-means (RQ-kmeans) to learn multi-level codes from item text features.
- **🔧 HSTU integration:** full pipeline to *replace* or *fuse* origina item ID embeddings with SID embeddings in various ways.
- **📊 Experiments & analysis:** ready-to-run experiments for ML-1M and ML-20M; tables & plots; ablations.
- **🎬 MovieLens enrichment:**  Extended MovieLens dataset with rich movie metadata and information and scripts to generate these items.



## Getting started

### Prerequisites
 
- First follow the environment setup from **[generative-recommenders](https://github.com/meta-recsys/generative-recommenders)**.

- Additional packages for SID & MovieLens enrichment:
    - `scikit-learn` (k‑means)
    - `transformers` (item text embeddings)
    - `kaggle` (Wikipedia plots dataset)
    - plus standard Python stack (numpy/pandas, etc.)

- Tested environment
    - Python 3.10(ubuntu22.04)
    - CUDA 11.8
    - Driver Version: 550.67
    - GPU: RTX 4090D

-  We provide an [environment.yml](./environment.yml) for your reference.

### Data Preparation

1. Run [preprocess_public_data.py](./preprocess_public_data.py) as described in [generative-recommenders](https://github.com/meta-recsys/generative-recommenders) to prepare movielens-1m or -20m.

2. Movielens extended information
    - Download directly: [Movielens-1m](https://drive.google.com/file/d/1viuDzrt-cQvz57nckEMJpO_4rAQEOqzd/view?usp=drive_link), [Movielens-20m](https://drive.google.com/file/d/1NX4A-AkcIQMiJjkOWJciTG2-AqIWbt7q/view?usp=drive_link)
    - Or rebuild them:

    ````bash
    export TMDB_READ_ACCESS_TOKEN="YOUR TMDB ACCESS TOKEN"
    export TMDB_KEY="YOUR TMDB KEY"
    python preprocess_movielens_extend.py \
        --dataset ml-1m \
        --movies_list tmp/processed/ml-1m/movies.csv \
        --save_file_pth ./tmp/processed/ml-1m/movie_extended_info.json
    ```` 

    - Outcome: should see the extended data files like: "./tmp/processed/ml-1m/movie_extended_info.json"

3. Movielens embeddings
    - Download directly: [Movielens-1m](https://drive.google.com/file/d/1m2shfZp7IvFieU63lDxnfgnro9jHAeWQ/view?usp=drive_link), [Movielens-20m](https://drive.google.com/file/d/1xjdpOTQ55JYhl4o1utchJIq4FkkCMlw8/view?usp=drive_link)
        - These were build with [qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Feel free to switch to any LLMs you prefer and rebuild.
    - Or rebuild hem:
        - Prerequisite: Should have the extended info in here: "./tmp/processed/ml-1m/movie_extended_info.json", otherwise will use the plain basic sentence.

    ```bash
    python preprocess_movielens_embed.py \
        --dataset ml-1m \
        --movies_list tmp/processed/ml-1m/movies.csv \
        --movies_info tmp/processed/ml-1m/movie_extended_info.json \
        --save_pth tmp/processed/ml-1m \
        --gpu_id 0 \
        --enc_name qwen3-0.6B \
        --enc_checkpoint Qwen/Qwen3-Embedding-0.6B \
        --max_sent_len 8192
    ```
    - Outcome: should have the embeddings files like "./tmp/processed/ml-1m/emb_ml-1m_qwen3-0.6B.pkl"

4. Semantic ID codebook and lookup table 
    - Download directly: [Movielens-1m codebook](https://drive.google.com/file/d/196vQPuhRkeei2M1SaYZOguRvKkU3VVFJ/view?usp=drive_link), [MovieLens-1m lookup table](https://drive.google.com/file/d/1-tnaDZWLSkTUzd2YcP6uIgKlAnvX_wU-/view?usp=drive_link), [Movielens-20m codebook](https://drive.google.com/file/d/12i5BgM5axcBL_K4LP0Q7Y9m7MXch3bAe/view?usp=drive_link), [MovieLens-20m lookup table](https://drive.google.com/file/d/1mYH7OlrAfERXbGr4x9uYjfJcsT-MClFJ/view?usp=drive_link)
        - These were built with 3 layers, 128 codes each, and 1 extra differentiation layer following google's [TIGER](https://papers.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf).
    - Or rebuild them:
        - Prerequisite: MUST have the embedding files ready like: "./tmp/processed/ml-1m/emb_ml-1m_qwen3-0.6B.pkl"

    ```bash
    python preprocess_SID.py \
        --data_pth "./tmp/processed/ml-1m/emb_ml-1m_qwen3-0.6B.pkl" \
        --movies_list "./tmp/processed/ml-1m/movies.csv" \
        --build_method "kmeans" \
        --num_layers 3 \
        --num_codes 128 \
        --seed 1024 \
        --add-uniq-layer \
        --save_pth "./tmp/processed/ml-1m" \
    ```

    - Outcome: should have the SID codebook files like: "./tmp/processed/ml-1m/SID_codebook_L3_C128_uniq.pkl" and lookup table files like: "./tmp/processed/ml-1m/SID_lookup_L3_C128_uniq.pkl".
    


### Training

#### Baseline HSTU

*Example*: default parameters on movielens-1m
```bash
DSET="ml-1m" &&
mkdir -p logs/${DSET}-l200 &&
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/${DSET}/hstu-sampled-softmax-n128-final.gin \
    --master_port=12345 \
    2>&1 | tee logs/${DSET}-l200/hstu-sampled-softmax-n128-final-BL.log
```

#### HSTU with SID
***REQUIRED***: must have the SID lookup tables and codebooks like: "the SID codebook files like: "./tmp/processed/ml-1m/SID_codebook_L3_C128_uniq.pkl" and lookup table files like: "./tmp/processed/ml-1m/SID_lookup_L3_C128_uniq.pkl".

*Example*: train HSTU on movielens-1m with 2-layer SID, embedding created with `prefixN` method, combined with the original item ID embedding with the basic `sum` fusion method.
```bash
DSET="ml-1m" &&
mkdir -p logs/${DSET}-l200 &&
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/${DSET}/hstu-sampled-softmax-n128-final-SID.gin \
    --master_port=12345 \
    2>&1 | tee logs/${DSET}-l200/hstu-sampled-softmax-n128-final-SID.log

```
For training with movielens-20m with 3 layers of SID (because 2 layers are not enough to cover ~28k items), run this config file: `configs/ml-20m/hstu-sampled-softmax-n128-final-SID.gin`.

**Notable SID config knobs**
There are several parameters that can be tuned to look at the effect of different setups. They can be change in the SID config gin files. Some important ones worth mentioning are as follows:

1. `SID_use_num_codebook_layers` – Use only the first min(*X*, SID_full_length) digits of semantic ID codes.
2. `SID_emb_method` – How to pool multi‑digit codes into one embedding for each item. Given a SID embedding table with *N* rows, C layers and K codes per layer
    - `Ngram`: row indexing = K^2 * c1 + K^1 * c2 + K^0 * c3 for a SID of (c1, c2, c3)
    - `prefixN`: obtain 3 rows: c1, K^1 * c1 + k^0 * c2, K^2 * c1 + K^1 * c2 + K^0 * c3 for a given SID of (c1, c2, c3), and then sum pooling all the corresponding embeddings.
    - `prefixN-indEmb`: We keep the original arbitrary ID embedding table, and fuse it with `prefixN` embeddings from the SID embedding table.
    - `RQsum-indEmb`: instead of *N*, we use *L * C* rows for the SID embedding table. We obtain 3 rows: c1, K * 1 + c1, K * 2 + c2 for a given SID of (c1, c2, c3), sum pooling and then fuse with the individual embedding
3. `SID_emb_comb_method` – fusion methods if we decide to fuse the SID embedding (E1) with the arbitrary ID embedding (E2): 
    - `sum`: sum(E1, E2)
    - `FC`: FC(concat(E1, E2))
    - `MLP`: FC(relu(FC(concat(E1, E2))))
    - `scalarGate`: gate (scalar) = sigmoid(FC(concat(E1, E2)))
    - `vectorGate`: gate (vector) = sigmoid(FC(concat(E1, E2)))
    - `scalarWeight`: learns alpha as a single parameter.


## Results & Analysis

Full results, analysis and discussion live in 👉 [my blog](https://shaox192.github.io/posts/2025/09/08/hstu-sid/).


## 🎬 Extended MovieLens Collection

One major contribution of this work is the systematic collection of extended movie metadata to enable meaningful SID generation:

- **Data Sources**: TMDB (via their official API), wikipedia movie plots (via this [kaggle dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)), IMDB (via their GraphQL API).
- **Features Collected**: Here is an example entry for the Toy story movie (items too long are replaced with "..."):
```json
{
    "1": {
            "overview": "Led by Woody, Andy's toys live happily in his room until ...",
            "budget": 30000000,
            "revenue": 394436586,
            "runtime": 81,
            "keywords": [
                "rescue",
                "friendship",
                "mission",
                "jealousy",
                "villain",
                ...
            ],
            "external_ids": {
                "imdb_id": "tt0114709",
                "wikidata_id": "Q171048",
                "facebook_id": null,
                "instagram_id": "toystory",
                "twitter_id": "toystory"
            },
            "release_date": "1995-11-22",
            "id": "tt0114709",
            "title": "Toy Story",
            "rating": 8.3,
            "vote_count": 1138304,
            "genres": [
                "Animation",
                "Adventure",
                "Comedy",
                "Family",
                "Fantasy"
            ],
            "imdb_plot": "A cowboy doll is profoundly jealous when ...",
            "languages": [
                "English"
            ],
            "countries": [
                "United States"
            ],
            "credits": {
                "Director": [
                    "John Lasseter"
                ],
                "Writers": [
                    "John Lasseter",
                    "Pete Docter",
                    "Andrew Stanton"
                ],
                "Stars": [
                    "Tom Hanks",
                    "Tim Allen",
                    "Don Rickles"
                ]
            },
            "technical_specs": {
                "sound_mixes": [
                    "Dolby Digital"
                ],
                "colorations": [
                    "Color"
                ]
            },
            "imdb_url": "https://www.imdb.com/title/tt0114709/",
            "similar_titles": [
                {
                    "id": "tt0435761",
                    "title": "Toy Story 3",
                    "year": 2010
                },
                {
                    "id": "tt0120363",
                    "title": "Toy Story 2",
                    "year": 1999
                },
                ...
            ],
            "featured_reviews": [
                {
                    "text": "This is a very clever animated story that was a big hit, and justifiably so. It had a terrific sequel and if a third film came out, that would probably be a hit, too.\n\nWhen this came out, ...",
                    "author_rating": 10
                },
                ...
            ]
        }
    ...
}
```

- **Quality Assurance**: some of the movies cannot be found automatically (or did not have a correct tmdb ID in the movielens-20m dataset), mostly because of different symbols in foreign languages or translation errors. I manually found these movies and saved their actual TMDB IDs [here](https://drive.google.com/file/d/1tpS275_L6Tz81QLrVhlv4cUK1pTrt2vJ/view?usp=drive_link). But this supplementary file should be automatically downloaded if rebuilding the extended information.


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{shao2025_hstu_sid,
  title  = {Enriching Generative Recommendation with Semantic IDs},
  author = {Zhenan Shao},
  year   = {2025},
  howpublished = {GitHub},
  url    = {https://github.com/shaox192/hstu-semantic-id}
}
```

## Contact

Feel free to email me @zhenans2@illinois.edu
