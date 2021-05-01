import pandas as pd
import zipfile
import os
import requests
import pickle

### Hyperparameters
RATING_THRESHOLD = 3
MIN_POSITIVE_RATINGS = 30

### Output
VERBOSE = 0

### URLS / File Paths
# downloads
LABELED_DATA = "https://conservancy.umn.edu/bitstream/handle/11299/198736/pair-responses.csv?sequence=9&isAllowed=y"
ML25_URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
# stored downloads
LABELED_DATA_STORED = "data/labeled.p1"
ML25_ZIP = "data/ml-25m.zip"
# processed data
RATINGS_1 = "data/processed/ratings_1.csv"
MOVIES_1 = "data/processed/movies_1.csv"


def load_movielens():
    os.makedirs("data/", exist_ok=True)

    if not os.path.isfile(ML25_ZIP):
        print("Downloading MovieLens dataset ...")
        movielens_downloaded_zip = requests.get(ML25_URL, stream=True)

        with open(ML25_ZIP, 'wb') as downloaded_file:
            downloaded_file.write(movielens_downloaded_zip.content)

        print("Downloaded MovieLens dataset")
        # unzipping file
        with zipfile.ZipFile(ML25_ZIP) as mlZip:
            print("Unzipping ML dataset ...")
            mlZip.extractall("data")
    else:
        print("Loaded MovieLens dataset")

    convert_to_binary()
    with zipfile.ZipFile(ML25_ZIP) as myzip:
        with myzip.open('ml-25m/ratings.csv') as ratings_f:
            ratings = pd.read_csv(ratings_f)
        with myzip.open('ml-25m/movies.csv') as movies_f:
            movies = pd.read_csv(movies_f)
    return ratings, movies


def load_labeled_dataset():
    if not os.path.isfile(LABELED_DATA_STORED):
        tmp = pd.read_csv(LABELED_DATA)
        pickle.dump(tmp, open(LABELED_DATA_STORED, "wb"))
        return tmp
    else:
        return pickle.load(open(LABELED_DATA_STORED, "rb"))


def get_ratings_movies(base_path):
    print("reading ratings csv ...")
    ratings = pd.read_csv(base_path + "ratings.csv")
    ratings["rating"] = (ratings["rating"] >= RATING_THRESHOLD).astype(int)
    ratings = ratings[ratings["rating"] == 1]

    print("reading movies csv ...")
    movies = pd.read_csv(base_path + "movies.csv")
    # movies.drop(["genres"], axis='columns', inplace=True)
    popularity = ratings[["movieId", "rating"]].groupby(["movieId"]).sum()

    merged = pd.merge(movies, popularity, left_on="movieId", right_on="movieId")
    movies = merged[merged["rating"] >= MIN_POSITIVE_RATINGS][["movieId", "title", "genres"]]

    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].dropna()

    if VERBOSE >= 1:
        print(ratings.head())
        print(movies.head())

    print(f"{len(ratings)} remaining ratings")
    print(f"{len(movies)} remaining movies")
    print(f"{len(ratings['userId'].unique())} remaining unique users")
    return ratings, movies


def convert_to_binary():
    if not (os.path.isfile(RATINGS_1) and os.path.isfile(MOVIES_1)):
        os.makedirs("data/processed/", exist_ok=True)

        ratings, movies = get_ratings_movies("data/ml-25m/")

        with open(RATINGS_1, "wb") as f:
            ratings.to_csv(f)

        with open(MOVIES_1, "wb") as f:
            movies.to_csv(f)

def get_reliable_responses(responses, ratings, movies):
    top_2500_ids = ratings.movieId.value_counts().index[:2500].tolist()
    inventory_ids = set(movies.movieId.tolist())
    responses["seedValid"] = responses.movieId.isin(top_2500_ids)
    responses["neighborValid"] = responses.neighborId.isin(inventory_ids)
    responses["valid"] = responses["seedValid"] & responses["neighborValid"]

    reliable_responses = responses[responses.valid].groupby(["movieId", "neighborId"]).filter(lambda x: len(x) >= 2)[
        ["movieId", "neighborId", "sim", "goodRec"]]
    return reliable_responses.groupby(["movieId", "neighborId"]).mean().sort_values(by=['movieId', 'sim'], ascending=[True, False])

def remove_rankings_with_not_enough_relevant_movies(reliable_responses):
    '''remove rankings without at least five relevant candidate movies'''
    def bin_by_relevance(x):
        bins = [-1, 2.0,  5]
        x["sim_bin"] = pd.cut(x["sim"], bins, labels=[0, 1], right=False)
        x["goodRec_bin"] = pd.cut(x["goodRec"], bins, labels=[0, 1], right=False)
        x["valid_group"] = (x["sim_bin"] == 1).sum() > 4
        return x

    diverse_rankings = reliable_responses.groupby("movieId").apply(bin_by_relevance)
    print("Before filtering out empty rankings:",
          diverse_rankings.reset_index()["movieId"].nunique())
    diverse_rankings = diverse_rankings[diverse_rankings.valid_group]
    print("After filtering out non-degenerate rankings:",
          diverse_rankings.reset_index()["movieId"].nunique())
    return diverse_rankings.reset_index()

if __name__ == '__main__':
    labeled = load_labeled_dataset()
    ratings, movies = load_movielens()

    reliable_responses = get_reliable_responses(labeled, ratings, movies)
    processed_rankings = remove_rankings_with_not_enough_relevant_movies(reliable_responses)

    print("\nLabeled set:")
    print(labeled.head())
