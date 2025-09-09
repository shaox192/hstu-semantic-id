# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Tuple
import os

import torch

from generative_recommenders.research.data.dataset import DatasetV2
from generative_recommenders.research.data.item_features import ItemFeatures
from generative_recommenders.research.data.preprocessor import get_common_preprocessors
# from data.semID_embedding import pickle_load
import pickle as pkl

def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)

@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    SID_codebook: torch.Tensor
    SID_lookup: torch.Tensor


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
    use_SID: bool = False,
    SID_data_pth_sfx: str = ""
) -> RecoDataset:
    if dataset_name == "ml-1m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
        )
    elif dataset_name == "ml-20m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif (
        dataset_name == "amzn-books"
    ):
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if dataset_name == "ml-1m" or dataset_name == "ml-20m":
        items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
        max_jagged_dimension = 16
        item_features: ItemFeatures = ItemFeatures(
            max_ind_range = [63, 16383, 511],
            num_items=dp.expected_max_item_id() + 1,
            max_jagged_dimension=max_jagged_dimension,
            lengths=[
                torch.zeros((dp.expected_max_item_id() + 1,), dtype=torch.int64),
                torch.zeros((dp.expected_max_item_id() + 1,), dtype=torch.int64),
                torch.zeros((dp.expected_max_item_id() + 1,), dtype=torch.int64),
            ],
            values=[
                torch.zeros((dp.expected_max_item_id() + 1, max_jagged_dimension), dtype=torch.int64),
                torch.zeros((dp.expected_max_item_id() + 1, max_jagged_dimension), dtype=torch.int64),
                torch.zeros((dp.expected_max_item_id() + 1, max_jagged_dimension), dtype=torch.int64),
            ],
        )
        all_item_ids = []
        for df_index, row in items.iterrows():
            #print(f"index {df_index}: {row}")
            movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")
            titles = row["cleaned_title"].split(" ")
            # print(f"{index}: genres{genres}, title{titles}")
            genres_vector = [hash(x) % item_features.max_ind_range[0] for x in genres]
            titles_vector = [hash(x) % item_features.max_ind_range[1] for x in titles]
            years_vector = [hash(row["year"]) % item_features.max_ind_range[2]]
            item_features.lengths[0][movie_id] = min(len(genres_vector), max_jagged_dimension)
            item_features.lengths[1][movie_id] = min(len(titles_vector), max_jagged_dimension)
            item_features.lengths[2][movie_id] = min(len(years_vector), max_jagged_dimension)
            for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
                for j in range(min(len(f_values), max_jagged_dimension)):
                    item_features.values[f][movie_id][j] = f_values[j]
            all_item_ids.append(movie_id)
        max_item_id = dp.expected_max_item_id()
        for x in all_item_ids:
            assert x > 0, "x in all_item_ids should be positive"
    else:
        # expected_max_item_id and item_features are not set for Amazon datasets.
        item_features = None
        all_item_ids = [x + 1 for x in range(dp.expected_num_unique_items())]
        max_item_id = dp.expected_num_unique_items()

    # Semantic ID loading
    sid_codebook, sid_lookup = None, None
    if use_SID:
        sid_codebook_f = f"./tmp/processed/{dataset_name}/SID_codebook_{SID_data_pth_sfx}.pkl"
        sid_lookup_f = f"./tmp/processed/{dataset_name}/SID_lookup_{SID_data_pth_sfx}.pkl"
        sid_codebook = pickle_load(sid_codebook_f)
        if sid_codebook is not torch.Tensor:
            sid_codebook = torch.cat([c.unsqueeze(0) for c in sid_codebook], dim=0)
        
        sid_lookup = None
        if os.path.exists(sid_lookup_f):
            sid_lookup = pickle_load(sid_lookup_f)
            if sid_lookup is not torch.Tensor:
                ## ! movie ids starts from 1, but are used directly in training and also there are skips in the middle
                ## number of skips are tiny, so able to just build a max size lookup table
                mov_ids = torch.tensor(list(sid_lookup.keys()), dtype=int)
                mov_sids = torch.tensor(list(sid_lookup.values()), dtype=int)
                
                sid_lookup = torch.zeros((torch.max(mov_ids).item() + 1, mov_sids.shape[1]), dtype=int)
                sid_lookup[torch.tensor(mov_ids, dtype=int)] = mov_sids + 1 if 0 in mov_sids else mov_sids
    
    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),
        max_item_id=max_item_id,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        SID_codebook=sid_codebook,
        SID_lookup=sid_lookup
    )
