"""
Search movie by IMDb ID and get detailed information

adapted from: https://github.com/pavan412kalyan/imdb-movie-scraper/blob/main/ImdbDataExtraction/search_by_id/search_movie.py
"""
import requests
import json

BASE_URL = "https://caching.graphql.imdb.com/"

HEADERS = {
    'accept': 'application/graphql+json, application/json',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://www.imdb.com',
    'priority': 'u=1, i',
    'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36'
}

def get_movie_details(movie_id):
    """Get detailed movie information by IMDb ID"""
    payload = {
        'query': """query GetTitle($id: ID!) {
          title(id: $id) {
            id
            titleText {
              text
            }
            ratingsSummary {
              aggregateRating
              voteCount
            }
            genres {
              genres {
                text
                id
              }
            }
            plot {
              plotText {
                plainText
              }
              language {
                id
              }
            }
            principalCredits {
              category {
                text
                id
              }
              credits {
                name {
                  nameText {
                    text
                  }
                }
                ... on Cast {
                  characters {
                    name
                  }
                }
                attributes {
                  text
                }
              }
            }
            spokenLanguages {
              spokenLanguages {
                text
                id
              }
            }
            countriesOfOrigin {
              countries {
                text
                id
              }
            }
            technicalSpecifications {
              soundMixes {
                items {
                  text
                }
              }
              colorations {
                items {
                  text
                }
              }
            }
            featuredReviews(first: 5) {
              edges {
                node {
                  text {
                    originalText {
                      plainText
                    }
                  }
                  authorRating
                }
              }
            }
            moreLikeThisTitles(first: 5) {
              edges {
                node {
                  id
                  titleText {
                    text
                  }
                  releaseYear {
                    year
                  }
                }
              }
            }
          }
        }""",
        'operationName': 'GetTitle',
        'variables': {
            'id': movie_id
        }
    }
    
    response = requests.post(BASE_URL, headers=HEADERS, json=payload)
    if response.status_code != 200:
        print(f"Response: {response.text}")
    response.raise_for_status()
    return response.json()

def format_movie_details(data):
    """Format movie details for display"""
    title_data = data.get("data", {}).get("title", {})
    if not title_data:
        return None
    
    # Basic info
    movie_id = title_data.get("id")
    title = title_data.get("titleText", {}).get("text")
     
    # Ratings
    ratings = title_data.get("ratingsSummary")
    rating = ratings.get("aggregateRating") if ratings else None
    vote_count = ratings.get("voteCount") if ratings else None

    # Genres
    genres = []
    genre_data = title_data.get("genres")
    if genre_data and isinstance(genre_data, dict):
        for genre in genre_data.get("genres", []):
            if genre and genre.get("text"):
                genres.append(genre.get("text"))
    
    # Plot
    plot_data = title_data.get("plot")
    plot = None
    if plot_data and isinstance(plot_data, dict):
        plot_text = plot_data.get("plotText")
        if plot_text and isinstance(plot_text, dict):
            plot = plot_text.get("plainText")

    # Credits
    credits_by_category = {}
    principal_credits = title_data.get("principalCredits", [])
    for credit_group in principal_credits:
        category = credit_group.get("category", {}).get("text", "Unknown")
        credits = credit_group.get("credits", [])
        
        category_credits = []
        for credit in credits:
            name_info = credit.get("name", {})
            name = name_info.get("nameText", {}).get("text")
            if name:
                category_credits.append(name)
        
        if category_credits:
            category_credits = category_credits[:3] # Limit to first 3 people per credit category
            credits_by_category[category] = category_credits
    
    # Languages
    languages = []
    lang_data = title_data.get("spokenLanguages")
    if lang_data and isinstance(lang_data, dict):
        for lang in lang_data.get("spokenLanguages", []):
            if lang and lang.get("text"):
                languages.append(lang.get("text")) 
    
    # Countries
    countries = []
    country_data = title_data.get("countriesOfOrigin")
    if country_data and isinstance(country_data, dict):
        for country in country_data.get("countries", []):
            if country and country.get("text"):
                countries.append(country.get("text"))
    
    # Technical specifications
    tech_specs = title_data.get("technicalSpecifications")
    sound_mixes = []
    colorations = []
    
    if tech_specs and isinstance(tech_specs, dict):
        sound_mix_data = tech_specs.get("soundMixes")
        if sound_mix_data and isinstance(sound_mix_data, dict):
            for item in sound_mix_data.get("items", []):
                if item and item.get("text"):
                    sound_mixes.append(item.get("text"))
        coloration_data = tech_specs.get("colorations")
        if coloration_data and isinstance(coloration_data, dict):
            for item in coloration_data.get("items", []):
                if item and item.get("text"):
                    colorations.append(item.get("text"))
      
    
    # featured reviews
    featured_reviews = []
    featured_rev_data = title_data.get("featuredReviews")
    if featured_rev_data and isinstance(featured_rev_data, dict):
        featured_rev_edges = featured_rev_data.get("edges", [])
        for edge in featured_rev_edges[:5]:  # Limit to first 5
            if edge and edge.get("node"):
                node = edge["node"]
                featured_reviews.append({
                    "text": node.get("text", {}).get("originalText", {}).get("plainText"),
                    "author_rating": node.get("authorRating")
                })
    
    
    # More like this titles
    similar_titles = []
    similar_data = title_data.get("moreLikeThisTitles")
    if similar_data and isinstance(similar_data, dict):
        similar_edges = similar_data.get("edges", [])
        for edge in similar_edges[:5]:  # Limit to first 5
            if edge and edge.get("node"):
                node = edge["node"]
                # print(node)
                similar_titles.append({
                    "id": node.get("id"),
                    "title": node.get("titleText", {}).get("text") if node.get("titleText", {}) else None,
                    "year": node.get("releaseYear", {}).get("year") if node.get("releaseYear", {}) else None,
                })
    
    
    return {
        "id": movie_id,
        "title": title,
        "rating": rating,
        "vote_count": vote_count,
        "genres": genres,
        "imdb_plot": plot,
        "languages": languages,
        "countries": countries,
        "credits": credits_by_category,
        "technical_specs": {
            "sound_mixes": sound_mixes,
            "colorations": colorations
        },
        "imdb_url": f"https://www.imdb.com/title/{movie_id}/" if movie_id else None,
        "similar_titles": similar_titles,
        "featured_reviews": featured_reviews
    }

def save_movie_data(movie_data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(movie_data, f, indent=4, ensure_ascii=False)
    print(f"Movie data saved to {filename}")

def get_imdb_movie_info(mov_id:str, save: bool=False):
    data = get_movie_details(mov_id)
    movie = format_movie_details(data)

    if save:
      filename = f"{movie['id']}_updated.json"
      save_movie_data(movie, filename)
    return movie
    
if __name__ == "__main__":
    # tests
    get_imdb_movie_info("tt0114709", save=True)
