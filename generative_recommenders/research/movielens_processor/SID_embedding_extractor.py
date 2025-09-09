
"""
generating content embeddings for items in the movie-lens dataset.
"""

# import argparse

import json
import os
from typing import List, Tuple, Dict
import torch

import numpy as np
import pandas as pd
import pickle as pkl

from transformers import AutoTokenizer, AutoModel


def pickle_dump(data, fpth: str):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)
        

def load_plm(model_path: str):

    if "qwen" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                   padding_side='left',
                                                   attn_implementation="flash_attention_2", 
                                                   torch_dtype=torch.float16)
        model = AutoModel.from_pretrained(model_path)  

    else:
        raise NotImplementedError(f"Only qwen model is supported now.")
        # tokenizer = AutoTokenizer.from_pretrained(model_path,)
        # model = AutoModel.from_pretrained(model_path,low_cpu_mem_usage=True,)
    return tokenizer, model


#### ===== movie description templates
## if no other info is available:
TEMPLATE_BASIC = "The movie titled: {}, shown in year {}, and belongs in the genre of: {}."
## if extended info is available:
TEMPLATE_DETAILED = """
Represent the movie {} ({}) with the following information for semantic understanding, clustering, and retrieval for recommendation to human viewers. There can be missing information.

Information:
- Plot summary: {};
- Genres: {};
- Keywords and Themes: {};
- Language: {};
- Country: {};
- IMDB recommended similar movies: {};
"""

# - Rating on IMDB: {};
# - Vote count on IMDB: {};
# - Language: {};
# - Country: {};
# - Runtime: {} minutes;
# - Director(s): {};
# - Writer(s): {};
# - Actor(s): {};
# - Featured reviews from IMDB users: {};


def _format_movie_info(title: str, yr: int, mov_info: Dict) -> str:
    credits = mov_info.get('credits', {})
    director, writer, actor = credits.get('Director', 'N/A'), mov_info.get('Writers', 'N/A'), mov_info.get('Stars', 'N/A')

    similar_movies = mov_info.get('similar_titles', [])
    similar_movies = [sm["title"] for sm in similar_movies]

    reviews = mov_info.get('featured_reviews', [])
    reviews = [f"Review {i} (author rating: {rv['author_rating']}): {rv['text']}; " for i, rv in enumerate(reviews)]

    desc = TEMPLATE_DETAILED.format(
        title, 
        yr,
        mov_info.get('overview', 'N/A'),
        mov_info.get('genres', 'N/A'),
        mov_info.get('keywords', 'N/A'),
    #   mov_info.get('rating', 'N/A'),
    #   mov_info.get('vote_count', 'N/A'),
        mov_info.get('languages', 'N/A'),
        mov_info.get('countries', 'N/A'),
    #   mov_info.get('runtime', 'N/A'),
    #   director, writer, actor,
        similar_movies, 
    #   reviews
    )

    return desc

def preprocess_text(args) -> Dict[int, str]:
    """
    Getting the list of movie items in movielens-1m or 20m, and convert them to descriptive text to be fed into LLMs.
    """
    movies_df = pd.read_csv(args.movies_list, delimiter=",",)
    
    movies_info = {}
    with open(args.movies_info, 'r') as f:
        movies_info = json.load(f)

    movie_desc_dict = {}
    for i, row in movies_df.iterrows():
        mov_id = int(row["movie_id"])
        title = row["cleaned_title"]
        yr = row["year"]
        if str(mov_id) not in movies_info:

            if len(movies_info)>0:
                print(f"!!Warning: No extended info found for movie_id {mov_id}, {title}, {yr}!! USING BASIC TEMPLATE.")

            genres = ', '.join(row["genres"].split('|'))
            desc = TEMPLATE_BASIC.format(title, yr, genres)
        else:
            curr_info = movies_info[str(mov_id)]
            desc = _format_movie_info(title, yr, curr_info)
        movie_desc_dict[mov_id] = desc
    
    print(f"... finished converting movie data, total: {len(movie_desc_dict)}, last entry: {movie_desc_dict[mov_id]}")
    return movie_desc_dict


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_item_embedding(args, item_text_dict, tokenizer, model) -> Dict[str, np.ndarray]:
    print(f'\n...Generate Text Embedding: ')

    mov_ids = np.array(list(item_text_dict.keys()))
    mov_text = list(item_text_dict.values())

    all_mov_embeddings = []
    start, batch_size = 0, 2
    with torch.no_grad():
        while start < len(item_text_dict):
            # if (start+1)%20==0:
            print(f"==> Embedding movies from #{start} to #{start+batch_size} out of {len(item_text_dict)}")
            curr_desc = mov_text[start : start + batch_size]

            # num_tokens = len(tokenizer.encode(curr_desc[0]))
            # print(f"num tokens: {num_tokens}")

            # Tokenize the input texts
            batch_dict = tokenizer(
                curr_desc,
                padding=True,
                truncation=True,
                max_length=args.max_sent_len,
                return_tensors="pt",
            )
            batch_dict.to(model.device)
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state,  # (batch_size, seq_len, hidden_size)
                                         batch_dict['attention_mask'] # (batch_size, seq_len)
                                         )
            del outputs
            unnorm_embeddings = embeddings.detach().cpu()
            # # normalize embeddings
            # embeddings = F.normalize(embeddings, p=2, dim=1)
            # scores = (embeddings[:2] @ embeddings[2:].T)

            all_mov_embeddings.append(unnorm_embeddings)
            start += batch_size
            torch.cuda.empty_cache()
            break

    all_mov_embeddings = torch.cat(all_mov_embeddings, dim=0).numpy()
    print(f"Finished generating embeddings: {all_mov_embeddings.shape}")

    results = {"movie_id": mov_ids,
               "embeddings": all_mov_embeddings}
    return results


def SID_embedding_extractor(args):
    item_text_dict = preprocess_text(args)

    plm_tokenizer, plm_model = load_plm(args.enc_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(args.device)
    print(f"\nLoaded pretrained llm ({args.enc_checkpoint}): ", plm_model)

    results = generate_item_embedding(args, item_text_dict, plm_tokenizer, plm_model)

    # output
    if args.save_pth is None:
        return
    
    if not os.path.exists(args.save_pth):
        os.makedirs(args.save_pth, exist_ok=True)
    file = os.path.join(args.save_pth, f"emb_{args.dataset}_{args.enc_name}.pkl")
    if args.save_suffix is not None:
        file = file.replace('.pkl', f'_{args.save_suffix}.pkl')
    pickle_dump(results, file)
    return
    



