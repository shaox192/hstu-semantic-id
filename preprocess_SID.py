
"""
main entry for building semantic ID from data representations and saving.
Built for movielens-1m and movielens-20m datasets. NOT for amazon books.

# movielens description here:https://github.com/manfredmichael/movielens-details

Usage example:

python preprocess_SID.py \
    --data_pth "./tmp/processed/ml-1m/emb_ml-1m_qwen3-0.6B.pkl" \
    --movies_list "./tmp/processed/ml-1m/movies.csv" \
    --build_method "kmeans" \
    --num_layers 2 \
    --num_codes 128 \
    --seed 1024 \
    --add-uniq-layer \
    --save_pth "./tmp/processed/ml-1m" \
"""


import os
import argparse
from typing import Dict, List, Tuple
import torch
import pandas as pd

import pickle as pkl

import numpy as np
from sklearn.cluster import KMeans


def make_parser():
    parser = argparse.ArgumentParser(description='Semantic ID builder')

    # files
    parser.add_argument('--data_pth', type=str, help='path to item representation data')
    parser.add_argument('--movies_list', type=str, 
                        help="[Optional for visualization purposes], path to the processed movie info file generated from preprocess_public_data.py" \
                        "for getting movie titles and genres etc.")

    # output
    parser.add_argument('--save_pth', type=str, help='path for saving the rawID --> SID lookup table')
    parser.add_argument('--save-suffix', type=str, default=None, help="unique suffix identifier to add to the saved files")
    
    # seed
    parser.add_argument('--seed', type=int, default=1024)

    # SID codebook parameters
    parser.add_argument('--build_method', default='kmeans', help='choose in [kmeans], ') #TODO: VAE? hierarchical kmeans?
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_codes', default=256, type=int) #TODO: non-uniform codes per layer?
    parser.add_argument('--add-uniq-layer', action='store_true', help="turn this on to add an additional layer to differentiate items. (TIGER style)")
    parser.add_argument('--norm-embs', action='store_true', help="normalize the item embeddings before building clustering.")
    return parser


def rq_kmeans(
        dat:np.ndarray, 
        dat_id: List, 
        num_layers: int, 
        num_codes: int, 
        seed:int = 1024
    ) -> Tuple[List, Dict, List]:
    
    """
    dat: should be training data, N * M
    dat_id: list of actual item ids, used as keys, len N
    num_layers: number of layers for the codebook
    num_codes: how many centers each layer of the codebook should have
    
    return:
        code_book_centers: list of tensors [num_layers * (num_codes * M)]
        SID_codes: numpy array of semantic ID (N * num_layers)
        kms_classifiers: list of KMeans objects in case the parameters are useful later
    """

    code_book_centers = []
    kms_classifiers = []
    SID_codes = {item_id: [] for item_id in dat_id}
    
    curr_residual = dat
    for l in range(num_layers):
        kms = KMeans(n_clusters=num_codes, random_state=seed, max_iter=300).fit(curr_residual)
        kms_classifiers.append(kms)
        
        code_book_centers.append(torch.tensor(kms.cluster_centers_))
        
        for j, item_id in enumerate(dat_id):
            SID_codes[item_id].append(kms.labels_[j].item())
        
        curr_residual -= kms.cluster_centers_[kms.labels_]
    
    return code_book_centers, SID_codes, kms_classifiers


def encode(codebook: List, item_data: torch.Tensor, classifiers: List = None) -> torch.Tensor:
    """
    Use codebook to encode item_data into semantic IDs.
    #TODO: INPROGRESS: not used yet.
    codebook: List of torch tensors
    item_data: torch tensor 
    """

    sem_ID = []
    for l_i in range(len(codebook)):
        if classifiers is not None:
            ids = classifiers[l_i].predict(item_data)
            sem_ID.append(torch.tensor(ids[:, None], dtype=torch.int64))
        else: # calc euclidean distance manually
            A = item_data
            B = codebook[l_i]
            
            X = A.norm(dim=-1).unsqueeze(-1) ** 2
            X = X.repeat(1, B.shape[0])
            Y = B.norm(dim=-1).unsqueeze(0)** 2
            Y = Y.repeat(A.shape[0], 1)
            euc_dist_mat = X + Y - 2 * A @ B.T
            out = torch.argmin(euc_dist_mat, dim=1)
            sem_ID.append(out[:, None])

    sem_ID = torch.cat(sem_ID, 1)
    return sem_ID


def build_codebook(
        embs_ids: List,
        embs: np.ndarray,
        codebook_build_method: str, 
        num_layers:int, 
        num_codes:int, 
        seed:int=1024, 
        add_uniq_layer: bool=False,
        norm_embs: bool = False
    ) -> Tuple[List, Dict]: 
    """
    embs_ids: list of item ids, len N
    embs: numpy array of item representations (movie embeddings), N * D    
    codebook_build_method: currently only "kmeans" is supported
    num_layers: number of layers for the codebook
    num_codes: how many centers each layer of the codebook should have
    seed: seed for kmeans
    add_uniq_layer: whether to add an additional layer to differentiate items with the same semantic ID (TIGER style)
    norm_embs: whether to normalize the embeddings before building the codebook

    return:
        codebook: cluster centers for each layer [num_layers * (num_codes * D)]
        SID_lookup: dict of actual movie ID --> SID codes
    """

    if norm_embs:
        embs_norm = np.linalg.norm(embs, ord=2, axis=1, keepdims=True)
        embs /= (embs_norm + 1e-12)
    
    lookup_tbl = codebook = None
    if codebook_build_method == "kmeans":
        codebook, lookup_tbl, kms_classifiers = rq_kmeans(embs, 
                                                          embs_ids, 
                                                          num_layers, 
                                                          num_codes, 
                                                          seed=seed)
    else:
        raise NotImplementedError

    if add_uniq_layer:
        print("Adding one more layer to distinguish among items with the same ID")
        curr_codes = np.asarray(list(lookup_tbl.values()), dtype=int)
        
        uniq_curr_codes, uniq_counts = np.unique(curr_codes, return_counts=True, axis=0)
        # print(uniq_curr_codes.shape, uniq_counts)
        print(f"max number of repetitions: {np.max(uniq_counts)}")
        uniq_dict = {tuple(code): cnt for (code, cnt) in zip(uniq_curr_codes, uniq_counts)}

        uniq_layer = np.zeros(curr_codes.shape[0], dtype=int)
        for i, item_code in enumerate(curr_codes):
            cnt = uniq_dict[tuple(item_code)]
            if cnt == 1: continue # already unique, just use 0
            uniq_layer[i] = cnt - 1
            uniq_dict[tuple(item_code)] -= 1
        
        for i, k in enumerate(lookup_tbl):
            # print(lookup_tbl[k])
            lookup_tbl[k] = [int(d) for d in (*curr_codes[i], uniq_layer[i])]
    
    return codebook, lookup_tbl


def _test_fake_data(N=10, D = 128):
    rng = np.random.default_rng(1024)
    arr1 = rng.random(size=(N, D))
    arr2 = rng.random(size=(N, D)) + 2
    arr3 = rng.random(size=(N, D)) + 3.5
    
    train = np.vstack([arr1, arr2, arr3])
    train_id = np.arange(train.shape[0], dtype = int)
    
    return train, train_id


def sid_inspect(SID, mov_df, num_codes):
    mov_sids = np.asarray(list(SID.values()),dtype=int)
    mov_orig_ids = np.asarray(list(SID.keys()))
    
    for i in range(num_codes): # 3):
        print(f"\n\n{i}")
        mask = mov_sids[:, 0] == i
        num_movs = np.sum(mask)
        movies = mov_orig_ids[mask]
        masked_movie_SIDs = mov_sids[mask]
        for j, mm in enumerate(movies):
            if j > 20: break
            print(masked_movie_SIDs[j],
                  mov_df[mov_df["movie_id"] == mm].iloc[0]["title"], 
                  mov_df[mov_df["movie_id"] == mm].iloc[0]["genres"],
                  )
        print(num_movs)

    return


def load_processed_embeddings(data_f: str):
    assert data_f.endswith('.pkl')
    with open(data_f, 'rb') as f:
        embs = pkl.load(f)
    return embs


def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def main(args):
    # if no embeddings file, make embeddings first, o/w load it
    if args.data_pth is None:
        print("No embs data, using fake data for testing.")
        embs, embs_id = _test_fake_data()
    else:
        item_embs = load_processed_embeddings(args.data_pth)
        embs, embs_id = item_embs["embeddings"], item_embs["movie_id"]
        print("Loaded item embeddings from ", args.data_pth)
        
    embs_id = embs_id.tolist() if embs_id is not List else embs_id

    # call build_codebook
    codebook, SID = build_codebook(
        embs_id,
        embs,
        args.build_method,
        args.num_layers,
        args.num_codes,
        args.seed,
        add_uniq_layer=args.add_uniq_layer,
        norm_embs=args.norm_embs,
    )

    ## save output
    if args.save_pth is not None:
        codebook_f = os.path.join(args.save_pth, f"SID_codebook_L{args.num_layers}_C{args.num_codes}")
        lookup_f = os.path.join(args.save_pth, f"SID_lookup_L{args.num_layers}_C{args.num_codes}")
        if args.save_suffix is not None:
            codebook_f += f"_{args.save_suffix}"
            lookup_f += f"_{args.save_suffix}"
        if args.norm_embs:
            codebook_f += "_normed"
            lookup_f += "_normed"
        if args.add_uniq_layer:
            codebook_f += "_uniq"
            lookup_f += "_uniq"
        codebook_f += ".pkl"
        lookup_f += ".pkl"
        print(f" Saving codebook to {codebook_f};\n Saving lookup table to {lookup_f}")
        pickle_dump(codebook, codebook_f)
        pickle_dump(SID, lookup_f)
    
    
    ### =========== sanity checks
    # example clusters
    if args.movies_list is not None:
        # load the title stuff for visualization purposes.
        movies_df = pd.read_csv(args.movies_list, delimiter=",",)
        sid_inspect(SID, movies_df, 5)

    # SID collisions
    USE_NUM_PREFIXES = 3
    HASH_TBL_SIZE = 294001 # 112909 # 2048 * 4 - 1

    SID = np.asarray(list(SID.values()), dtype=int)
    print("SID shape: ", SID.shape)
    print("num unique SID tuple: ", np.unique(SID, axis=0).shape)

    SID = SID[:, :USE_NUM_PREFIXES]
    print(f"num unique SID tuple with prefix {USE_NUM_PREFIXES}: ", np.unique(SID, axis=0).shape)

    convert_coeff = [args.num_codes ** i for i in range(SID.shape[-1])]
    convert_coeff.reverse()
    print("Ngram conversion coeff: ", convert_coeff)
    convert_coeff = np.asarray(convert_coeff)
    convert_sid = SID * convert_coeff
    print(SID[0, :], convert_sid[0, :])
    
    emb_id = np.sum(convert_sid, axis = -1)
    print("Ngram converted num unique items: ", np.unique(emb_id).shape)

    emb_tbl_idx = np.mod(emb_id, HASH_TBL_SIZE)
    print(f"Ngram converted hashed with {HASH_TBL_SIZE}: {np.unique(emb_tbl_idx).shape}")
    return


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    
    print(vars(args))

    main(args)


