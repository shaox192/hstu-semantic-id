"""
Analyze the distribution of unique SID IDs, prefixes in user histories.

#TODO: save and plot in a separate ipynb
"""

import random

import os

import torch
import pandas as pd
import pickle as pkl
import numpy as np


def main(
    random_seed: int = 42,
) -> None:
    
    ########## Set datasets and files to process:
    dataset = "ml-20m"
    ratings_file_pth = f"./tmp/{dataset}/sasrec_format.csv"
    sid_lookup_pth = f"./tmp/processed/{dataset}/SID_lookup_L3_C128_uniq.pkl"
    NUM_LAYERS = 4
    ##########

    # to enable more deterministic results.
    random.seed(random_seed)

    ratings_df = pd.read_csv(
            ratings_file_pth,
            delimiter=",",
        )
    print(f"ratings_df shape: {ratings_df.shape}")

    sid_lookup = pkl.load(open(sid_lookup_pth, "rb"))
    print(f"len sid_lookup: {len(sid_lookup)}")

    mov_ids = torch.tensor(list(sid_lookup.keys()), dtype=int)
    mov_sids = torch.tensor(list(sid_lookup.values()), dtype=int)
    print(max(mov_ids), min(mov_ids))

    sid_lookup = torch.zeros((torch.max(mov_ids).item() + 1, mov_sids.shape[1]), dtype=int)
    sid_lookup[torch.tensor(mov_ids, dtype=int)] = mov_sids + 1 if 0 in mov_sids else mov_sids


    user_hist_uniq_cnts_singledigit = [[] for _ in range(NUM_LAYERS)]
    user_hist_uniq_cnts_prefix = [[] for _ in range(NUM_LAYERS)]
    for i, row in ratings_df.iterrows():
        if i % 10000 == 0:
            print(f" -- processing user {i} / {len(ratings_df)}")

        curr_seq = row.sequence_item_ids
        curr_seq = eval(curr_seq)
        curr_seq = [curr_seq] if type(curr_seq) == int else list(curr_seq)

        curr_sid_seq = [sid_lookup[x].tolist() for x in curr_seq]
        # print([c[2] for c in curr_sid_seq])

        for p_i in range(NUM_LAYERS):
            prefix_len = p_i + 1

            # find unique single digits
            sid_singledigits = [sid[p_i] for sid in curr_sid_seq]
            sid_singledigits_uniq = set(sid_singledigits)
            user_hist_uniq_cnts_singledigit[p_i].append((len(sid_singledigits_uniq), len(sid_singledigits)))

            # find unique prefixes
            sid_prefixes = [tuple(sid[:prefix_len]) for sid in curr_sid_seq]
            sid_prefixes_uniq = set(sid_prefixes)
            user_hist_uniq_cnts_prefix[p_i].append((len(sid_prefixes_uniq), len(sid_prefixes)))
    # exit()
    user_hist_uniq_cnts_singledigit = np.array(user_hist_uniq_cnts_singledigit)
    user_hist_uniq_cnts_prefix = np.array(user_hist_uniq_cnts_prefix)
    
    full_uniq_cnts_singledigit = []
    full_uniq_cnts_prefix = []
    for i in range(NUM_LAYERS):
        curr_uniq_cnts = user_hist_uniq_cnts_prefix[i]
        uniq_percent = curr_uniq_cnts[:, 0] / curr_uniq_cnts[:, 1]
        full_uniq_cnts_prefix.append(uniq_percent)

        curr_uniq_cnts = user_hist_uniq_cnts_singledigit[i]
        uniq_percent = curr_uniq_cnts[:, 0] / curr_uniq_cnts[:, 1]
        full_uniq_cnts_singledigit.append(uniq_percent)
    
    save_dict = {
        "full_uniq_cnts_singledigit": full_uniq_cnts_singledigit,
        "full_uniq_cnts_prefix": full_uniq_cnts_prefix,
    }
    save_pth = f"./analysis/output/SID_distribution_{dataset}.pkl"
    pkl.dump(save_dict, open(save_pth, "wb"))
    print(f"Distribution saved to {save_pth}")
    


if __name__ == "__main__":
    main()


