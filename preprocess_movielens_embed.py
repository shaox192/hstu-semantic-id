"""
Main entry point for generating content embeddings for items in the movie-lens dataset.

*** Need the processed data ready from: 

1. preprocess_public_data.py --> "tmp/processed/ml-1m/movies.csv"
2. movielens_extend.py --> "tmp/processed/ml-1m/movie_extended_info.json"

before running this script ***

example usage:

python preprocess_movielens_embed.py \
    --dataset ml-1m \
    --movies_list tmp/processed/ml-1m/movies.csv \
    --movies_info tmp/processed/ml-1m/movie_extended_info.json \
    --save_pth tmp/processed/ml-1m \
    --gpu_id 0 \
    --enc_name qwen3-0.6B \
    --enc_checkpoint Qwen/Qwen3-Embedding-0.6B \
    --max_sent_len 8192
    
"""

import argparse
import torch
from generative_recommenders.research.movielens_processor.SID_embedding_extractor import SID_embedding_extractor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help = "name of the dataset: [ml-1m], [ml-20m]")
    parser.add_argument('--movies_list', type=str, required=True, help="item pool: path to the processed movie info file generated from preprocess_public_data.py")
    parser.add_argument('--movies_info', type=str, help="movie descriptions and info")

    parser.add_argument('--save_pth', type=str, help="where to save the final representation file")
    parser.add_argument('--save-suffix', type=str, default=None, help="unique suffix identifier to add to the saved files")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    
    # default parameters for using qwen3-0.6B
    parser.add_argument('--enc_name', type=str, default='qwen3-0.6B')
    parser.add_argument('--enc_checkpoint', type=str, default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--max_sent_len', type=int, default=8192)
    # parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    device = "cpu" if args.gpu_id < 0 else set_device(args.gpu_id)
    args.device = device

    SID_embedding_extractor(args)