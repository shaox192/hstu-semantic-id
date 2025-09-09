
"""
extend the movielens movie information by querying TMDB, IMDB, and Wikipedia.
"""

import json
import os
import requests

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import time

from generative_recommenders.research.movielens_processor.imdb_scraper import get_imdb_movie_info



def movie_list_loader(args):
    movies_df = pd.read_csv(args.movies_list, delimiter=",",)

    movie_desc_dict = {}
    for i, row in movies_df.iterrows():
        mov_id = int(row["movie_id"])
        title = row["cleaned_title"]
        # print(title)
        if title.endswith('The'):
            title = 'The ' + title[:-len(", The")]
        yr = row["year"]
        genres = row["genres"].split('|')
        movie_desc_dict[mov_id] = [title, yr, genres]

    print(f"... finished loading movie data, total: {len(movie_desc_dict)}, last entry: {movie_desc_dict[mov_id]}")
    return movie_desc_dict


def tmdb_search_movie(title: str, year: int, SESSION):
    SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

    titles = []
    if "a.k.a." in title:
        t1, t2 = title.split("(a.k.a.")[:2]
        t1 = t1.strip()
        t2 = t2.strip().strip(")").strip()
        titles = [t1, t2]
    elif '(' in title and title.endswith(')'):
        t1 = title[:title.rfind('(')].strip()
        t2 = title[title.rfind('(')+1:-1].strip()
        titles = [title, t1, t2]
    else:
        titles = [title]
    # print(title, year)
    for t in titles:
        t = t.replace(" ", "%20")
        t = t.replace("'", "%27")
        url = f"{SEARCH_URL}?query={t}&include_adult=false" # &year={year}"
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        # print(r.json())
        results = r.json().get("results", [])

        if not results:
            continue
        if len(results) == 1:
            return results[0]

        exact = [m for m in results if (m.get("title","").lower()==t.lower() and
                                        str(m.get("release_date",""))[:4]==str(year))]
        cand = exact or results
        return sorted(cand, key=lambda m: (m.get("vote_count",0), m.get("popularity",0)), reverse=True)[0]
    return None


def tmdb_movie_details_cleanup(details_json:str) -> Dict:

    details = {
        "overview": details_json.get("overview", ""),
        "budget": details_json.get("budget", 0),
        "revenue": details_json.get("revenue", 0),
        "runtime": details_json.get("runtime", 0),
        # "country": [cc['name'] for cc in details_json.get("production_countries", [])],
        # "language": [ll['name'] for ll in details_json.get("spoken_languages", [])],
        "keywords": [kk['name'] for kk in details_json.get("keywords", {}).get("keywords", [])],
        "external_ids": details_json.get("external_ids", {}),
        "release_date": details_json.get("release_date", None),
        }
        
    return details

def tmdb_get_movie_details(tmdb_id: str, SESSION,query_props: List[str]=None):
    ## TMDB details query
    DETAIL_URLS = ["https://api.themoviedb.org/3/movie",
                "https://api.themoviedb.org/3/tv"]
    QUERY_PROPERTIES = ["overview", "runtime", "budget", "revenue", "keywords", "external_ids"] # , "credits", "reviews", 
    query_props = query_props or QUERY_PROPERTIES

    query_str = '%2C'.join(query_props)
    for url in DETAIL_URLS:
        url = f"{url}/{tmdb_id}?append_to_response={query_str}"
        r = SESSION.get(url, timeout=20)
        if "<Response [404]>" in str(r):
            continue
        else:
            break
    if "<Response [404]>" in str(r):
        return {}
    r.raise_for_status()
    out_dict = tmdb_movie_details_cleanup(r.json())
    out_dict["tmdb_id"] = tmdb_id
    return out_dict


def get_wikidata_movie_plot(wiki_df: pd.DataFrame, title: str, yr:str) -> str:
    """
    Got this from here: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?resource=download
    """

    mov_row = wiki_df[wiki_df["Title"].str.lower() == title.lower().strip()]
    if mov_row.empty:
        return ""
    if len(mov_row) > 1:
        print(f"\t[WIKI]! Warning: found multiple movies with the same title: {title}, matching the year: {yr}")
        mov_row = mov_row[mov_row["Release Year"] == yr]
        if mov_row.empty:
            print(f"\t[WIKI]! Warning: cannot find the movie with the same year: {yr}, skipping ...")
            return ""
        if len(mov_row) > 1:
            print(f"\t[WIKI]! Warning: still found multiple movies with the same title and year: {title} ({yr}), taking the first one.")

    return mov_row.iloc[0]["Plot"]


def movielens_extender(args):

    ## load movie list
    item_text_dict = movie_list_loader(args)
    ## TMDB API setup
    SESSION = requests.Session()
    SESSION.headers.update({"Accept": "application/json",
                            "Authorization": f"Bearer {args.TMDB_READ_ACCESS_TOKEN}"})


    ## These are manually searched and verified tmdb ids for some movies with difficult titles.
    tmdb_title2ID = {}
    if os.path.exists(args.tmdb_manual_ids_fpth):
        tmdb_title2ID = json.load(open(args.tmdb_manual_ids_fpth, "r"))

    ## wikipedia movie plots
    wiki_df = pd.read_csv(args.wiki_mov_pth,header=0)

    ## load previously saved ml-1m details to help lookup for ml-20m
    ml_1m_details = {}
    ml_1m_imdb_ids = {}
    if (args.dataset == "ml-20m") and (args.prev_movies_info is not None):
        ml_1m_details = json.load(open(args.prev_movies_info, "r"))
        ml_1m_imdb_ids = {}
        for k, v in ml_1m_details.items():
            imdb_id = v.get("external_ids", {}).get("imdb_id", None)
            if imdb_id is not None:
                ml_1m_imdb_ids[imdb_id] = k
        print(f"loaded {len(ml_1m_details)} ml-1m movies from {args.prev_movies_info}")

    ## load links file if ml-20m
    movies_known_id2tmdbid = {}
    if (args.dataset == "ml-20m") and (args.links_file is not None):
        links_df = pd.read_csv(args.links_file)
        for i, row in links_df.iterrows():
            curr_mid = str(int(row["movieId"]))
            curr_tmdb_id = row["tmdbId"]
            if not np.isnan(curr_tmdb_id):
                movies_known_id2tmdbid[curr_mid] = str(int(curr_tmdb_id))
        print(f"loaded {len(movies_known_id2tmdbid)} known links from {args.links_file}")

    ## whether to continue from existing save file
    existing_movie_details = {}
    if args.continue_from_save and os.path.exists(args.save_file_pth):
        print(f"continue from existing file: {args.save_file_pth}")
        existing_movie_details = json.load(open(args.save_file_pth, "r"))
        print(f"loaded {len(existing_movie_details)} movies from existing file.")

    all_mov_details = {}
    for cnt, (id, v) in enumerate(item_text_dict.items()):
        if cnt > 35: break
        if str(id) in existing_movie_details:
            all_mov_details[id] = existing_movie_details[str(id)]
            continue
        if (cnt != 0) and (cnt % 30 == 0):
            print(f"\n==> processing [{cnt + 1}]/[{len(item_text_dict)}], movie id: {id}, saving and sleeping for 3 seconds ...", flush=True)
            with open(args.save_file_pth, "w") as f:
                f.write(json.dumps(all_mov_details, indent=4))

            time.sleep(5) # to avoid getting 429
            print(f"\n... resume processing ...", flush=True)

        # search for tmdb movie ids
        title, yr, genres = v
        if title in tmdb_title2ID:
            tmdb_id = tmdb_title2ID[title]
        else:
            movie_data = tmdb_search_movie(title, yr, SESSION)
            tmdb_id = movie_data.get("id", None) if movie_data is not None else None

        if (tmdb_id is None) and (str(id) in movies_known_id2tmdbid):
                tmdb_id = movies_known_id2tmdbid[str(id)]
        
        if tmdb_id is None:
            print(f"!![TMDB]! Cannot find movie: ml id: {id}, title: {title}, year: {yr}")
            continue

        ## getting tmdb details
        curr_mov_details = tmdb_get_movie_details(tmdb_id, SESSION)
        if len(curr_mov_details) == 0: # this would be invalid tmdb id situation
            print(f"!![TMDB]! Cannot get details for movie: ml id: {id}, title: {title}, year: {yr}, tmdb id: {tmdb_id}")
            continue
        imdb_id = curr_mov_details.get("external_ids", {}).get("imdb_id", None)
        if imdb_id is not None and (imdb_id in ml_1m_imdb_ids):            
            all_mov_details[id] = ml_1m_details[ml_1m_imdb_ids[imdb_id]]
            continue

        ## wikipedia movie plots
        wiki_plot = get_wikidata_movie_plot(wiki_df, title, yr)

        ## imdb reviews and details
        extern_id = curr_mov_details.get("external_ids", {})
        imdb_id = extern_id.get("imdb_id", None)
        imdb_infos = None
        if imdb_id:
            imdb_infos = get_imdb_movie_info(imdb_id)

        ## aggregate all info
        #change the overview to the longest one among tmdb, imdb, wiki
        all_mov_details[id] = curr_mov_details
        tmdb_plot = curr_mov_details.get("overview","")
        imdb_plot = imdb_infos.get("imdb_plot","") if imdb_infos is not None else ""
        imdb_plot = imdb_plot if imdb_plot is not None else ""

        all_mov_details[id]["overview"] = max([tmdb_plot, imdb_plot, wiki_plot], key=len)
        
        #update imdb_infos
        if imdb_infos is not None:
            all_mov_details[id].update(imdb_infos)
    
    # wrap up
    if args.save_file_pth is None:
        return
    with open(args.save_file_pth, "w") as f:
        f.write(json.dumps(all_mov_details, indent=4))


def _download_from_gdrive(url: str, output_file: str):
    response = requests.get(url, allow_redirects=True)

    with open(output_file, "wb") as f:
        f.write(response.content)


def setup(tmdb_manual_ids_fpth, wiki_mov_pth):
    # check tmdb api keys exists:
    assert "TMDB_READ_ACCESS_TOKEN" in os.environ, "Missing TMDB_READ_ACCESS_TOKEN environment variable!"
    assert "TMDB_KEY" in os.environ, "Missing TMDB_KEY environment variable!"
    
    if not os.path.exists(tmdb_manual_ids_fpth):
        print("... downloading the ml_tmdb_name2id.json file ...")
        url = "https://drive.usercontent.google.com/download?id=1tpS275_L6Tz81QLrVhlv4cUK1pTrt2vJ&confirm=t"
        output_file = tmdb_manual_ids_fpth
        _download_from_gdrive(url, output_file)
        
    if not os.path.exists(wiki_mov_pth):
        print("... downloading the wiki_movie_plots_deduped.csv file from kaggle ...")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("jrobischon/wikipedia-movie-plots", 
                                   path=os.path.sep.join(wiki_mov_pth.split(os.path.sep)[:-1]), 
                                   unzip=True)

        


