"""
Bin the movies based on the frequency of appearances.
"""

import random

import pandas as pd
import pickle as pkl
import numpy as np


BIN_THRES = [0.25, 0.75]  # cumulative frequency bins: 25%, 75%

def main(
    random_seed: int = 42,
) -> None:
    
    ########## Set datasets and files to process:
    dataset = "ml-20m"
    ratings_file_pth = f"./tmp/{dataset}/sasrec_format.csv"
    movie_items_file_pth = f"./tmp/processed/{dataset}/movies.csv"
    ##########

    # to enable more deterministic results.
    random.seed(random_seed)

    ratings_df = pd.read_csv(
            ratings_file_pth,
            delimiter=",",
        )
    print(f"ratings_df shape: {ratings_df.shape}")
    print(ratings_df.head())
    # exit(0)

    mov_items_df = pd.read_csv(
            movie_items_file_pth,
            delimiter=",",
        )
    print(f"mov_items_df shape: {mov_items_df.shape}")
    print(mov_items_df.head())

    all_IDs = mov_items_df.movie_id.tolist()
    print(f"len all_IDs: {len(all_IDs)}, min: {min(all_IDs)}, max: {max(all_IDs)}, type: {type(all_IDs[0])}")
    
    frequency_cnt = {k: 0 for k in all_IDs}
    for i, row in ratings_df.iterrows():
        if i % 10000 == 0:
            print(f" -- processing user {i} / {len(ratings_df)}")

        curr_seq = row.sequence_item_ids
        curr_seq = eval(curr_seq)
        curr_seq = [curr_seq] if type(curr_seq) == int else list(curr_seq)

        for curr_id in curr_seq:
            assert curr_id in frequency_cnt, f"curr_id {curr_id} not in frequency_cnt"
            frequency_cnt[curr_id] += 1

    sorted_freq_cnt = sorted(frequency_cnt.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 10 frequent items: {sorted_freq_cnt[:10]}")
    
    # bin by cumulative frequency: 25%, 50%, 75%, 100%
    freq_cnt_arr = np.array([x[1] for x in sorted_freq_cnt], dtype=np.float32)
    freq_cnt_arr /= freq_cnt_arr.sum()
    print(f"Frequency counts sum to 1?: {freq_cnt_arr.sum()}")
    
    cum_freq = np.cumsum(freq_cnt_arr)
    bin_indices = np.zeros(len(cum_freq), dtype=np.int32)
    for i, thres in enumerate(BIN_THRES):
        bin_indices += (cum_freq >= thres).astype(np.int32)
    print(f"Bin indices: {bin_indices}")

    movie_id_2_bins = {x[0]: bin_indices[i] for i, x in enumerate(sorted_freq_cnt)}

    sorted_freq_cnt_dict = dict(sorted_freq_cnt)
    save_dict = {
        "sorted_freq_cnt_dict": sorted_freq_cnt_dict,
        "cumulative_frequency": cum_freq,
        "movie_id_2_bins": movie_id_2_bins,
    }
    save_pth = f"./analysis/movie_binning_{dataset}.pkl"
    pkl.dump(save_dict, open(save_pth, "wb"))
    print(f"Distribution saved to {save_pth}")
    


if __name__ == "__main__":
    main()
