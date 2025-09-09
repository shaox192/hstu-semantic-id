"""
Main entry point for extending the movielens movie information by querying TMDB, IMDB, and Wikipedia.

*** Need the processed data ready from: 

    1. preprocess_public_data.py --> "tmp/processed/ml-1m/movies.csv"

before running this script ***

This script will download 2 additional files into ./tmp/ if not already present:
    manually found tmdb ids for some movies with difficult names "./tmp/ml_tmdb_name2id.json"
    wikipedia movie plots"./tmp/wiki_movie_plots_deduped.csv" --> from kaggle.
and then write the final extended movie information to:
    --save_file_pth ./tmp/processed/ml-1m/movie_extended_info.json

Will retrived these information for a movie if available: 
    "overview": the movie plot, will take the longest one from TMDB or Wikipedia or IMDB,
    "budget": budget of the movie
    "revenue": revenue of the movie,
    "runtime": runtime of the movie,
    "keywords": keywords associated with the movie as a list,
    "external_ids": useful ids for other scraping like the imdb_id (e.g., "tt0114709" for Toy Story),
    "release_date": release date,
    "tmdb_id": the tmdb id,
    "id": this it the IMDB id
    "title": the movie title,
    "rating": IMDB rating score,
    "vote_count": IMDB vote count,
    "genres": IMDB genres,
    "imdb_plot": still saving the IMDB plot
    "languages": languages the movie is in,
    "countries": countries the movie is from,
    "credits": directors, writers, and actors (top 3)
    "technical_specs": sound_mixes liks is it dolby, colorations like is it color or black and white, etc.
    "imdb_url": the url to the imdb page,
    "similar_titles": imdb's recommendations for similar movies: top5
    "featured_reviews": top5 imdb reviews, with both the content and the rating score if available.

** example usage:

export TMDB_READ_ACCESS_TOKEN="YOUR_TMDB_READ_ACCESS_TOKEN"
export TMDB_KEY="YOUR_TMDB_KEY"

python preprocess_movielens_extend.py \
    --dataset ml-1m \
    --movies_list tmp/processed/ml-1m/movies.csv \
    --save_file_pth ./tmp/processed/ml-1m/movie_extended_info.json
    
"""

import argparse
import torch
import os
import requests
import json

from generative_recommenders.research.movielens_processor.movielens_extender import movielens_extender, setup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ml-1m", help = "name of the dataset")
    parser.add_argument('--movies_list', type=str, default="./tmp/processed/ml-1m/movies.csv", help="path to the processed movie info file generated from preprocess_public_data.py")
    parser.add_argument('--save_file_pth', type=str, default="./tmp/processed/ml-1m/movie_extended_info.json", help="final representation will be saved to this")

    parser.add_argument('--continue_from_save', action='store_true', help="whether to continue from the existing save_file_pth, this helps if the scraper got interrupted in the middle")

    # only useful for ml-20m
    parser.add_argument('--prev_movies_info', type=str, default="./tmp/processed/ml-1m/movie_extended_info.json", help="path to ml-1m details json, can be used to lookup for [ml-20m]")
    parser.add_argument('--links_file', type=str, default="./tmp/ml-20m/links.csv", help="path to the links.csv file if using [ml-20m], this helps to lookup known tmdb ids, although not all movies have valid tmdb ids.")
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

    tmdb_manual_ids_fpth = "./tmp/ml_tmdb_name2id.json"
    wiki_mov_pth = "./tmp/wiki_movie_plots_deduped.csv"
    setup(tmdb_manual_ids_fpth, wiki_mov_pth)

    args.tmdb_manual_ids_fpth = tmdb_manual_ids_fpth
    args.wiki_mov_pth = wiki_mov_pth
    args.TMDB_READ_ACCESS_TOKEN = os.environ.get("TMDB_READ_ACCESS_TOKEN")
    args.TMDB_KEY = os.environ.get("TMDB_KEY")

    movielens_extender(args)
