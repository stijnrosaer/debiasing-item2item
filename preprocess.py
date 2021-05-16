import pandas as pd
import numpy as np
import zipfile
import os
import requests
import pickle
import scipy.sparse as sp
from datetime import datetime

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
RATINGS_COO_1 = "data/processed/ratings_coo_1.csv"
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
        print("Loading Labeled Dataset ...")
        tmp = pd.read_csv(LABELED_DATA)
        pickle.dump(tmp, open(LABELED_DATA_STORED, "wb"))
        return tmp
    else:
        print("Loaded Labeled Dataset")
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
    movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').fillna("-1").astype(int)



    ratings = ratings[ratings["movieId"].isin(movies["movieId"])].dropna()

    i = ratings["movieId"].values
    j = ratings["userId"].values

    ratings_coo = sp.coo_matrix((ratings["rating"].values, (i, j))).transpose()

    remaining_movie_ids = np.unique(ratings_coo.col)
    new_ids = -np.ones(ratings_coo.shape[1], dtype=int)
    new_ids[remaining_movie_ids] = np.arange(remaining_movie_ids.size)

    ratings_coo.col = new_ids[ratings_coo.col]

    movies["newId"] = new_ids[movies.movieId]

    t = ratings.groupby("movieId").min()["timestamp"].reset_index()

    def convert(row):
        return datetime.fromtimestamp(row).year

    t["timestamp"] = t["timestamp"].apply(convert)

    print(t.head())
    movies["year_first"] = movies["movieId"].map(t.set_index("movieId")["timestamp"])


    ratings.drop(["timestamp"], axis="columns", inplace=True)
    ratings_coo = ratings_coo.tocoo()

    if VERBOSE >= 1:
        print(ratings.head())
        print(movies.head())

    print(f"{len(ratings)} remaining ratings")
    print(f"{len(movies)} remaining movies")
    print(f"{len(ratings['userId'].unique())} remaining users")

    ratings_data = np.hstack((ratings_coo.row.reshape(-1, 1), ratings_coo.col.reshape(-1, 1),
                        ratings_coo.data.reshape(-1, 1)))

    np.savetxt(RATINGS_COO_1, ratings_data, fmt="%d", delimiter=",")

    return ratings, movies


def convert_to_binary():
    if not (os.path.isfile(RATINGS_1) and os.path.isfile(MOVIES_1)):
        os.makedirs("data/processed/", exist_ok=True)

        ratings, movies = get_ratings_movies("data/ml-25m/")

        with open(RATINGS_1, "wb") as f:
            ratings.to_csv(f, index=False)

        with open(MOVIES_1, "wb") as f:
            movies.to_csv(f)


def split_relevant_judgements(labeled, ratings, movies):
    # remove deleted movies
    movieIds = movies["movieId"].tolist()
    labeled = labeled[labeled["movieId"].isin(movieIds)]
    labeled = labeled[labeled["neighborId"].isin(movieIds)]
    labeled.drop(["userId"], axis='columns', inplace=True)

    # Judgements van min 2 mensen, bin op average score
    labeled = labeled.groupby(["movieId", "neighborId"]).filter(lambda x: len(x) >= 2)
    labeled = labeled.groupby(["movieId", "neighborId"]).mean()

    # avg score > 3 = relevant
    labeled["sim_bin"] = (labeled["sim"] >= 2.0).astype(int)

    # enkel films met 4 positive pairs --> 67 films over in paper
    popularity = labeled.groupby(["movieId"]).sum()
    popularity = popularity[popularity["sim_bin"] > 4]
    labeled = labeled.reset_index()
    popularity = popularity.reset_index()
    labeled = labeled[labeled["movieId"].isin(popularity["movieId"])].dropna()

    # split into test, train and val set
    permutation = np.random.permutation(labeled["movieId"].unique())
    train_size = int(len(permutation) * 0.5)
    test_size = int(len(permutation) * 0.25)
    val_size = test_size

    splits = dict(zip(['train', 'val', 'test'], np.split(permutation, [train_size, train_size + val_size])))

    result = {}

    os.makedirs("datasets/labeled/", exist_ok=True)
    for name, index in splits.items():
        split = labeled[labeled["movieId"].isin(index)]
        result[name] = split
        split.to_csv("data/labeled/" + name + ".csv", index=False)

    print("train size:", result["train"]["movieId"].nunique())
    print("validation size:", result["val"]["movieId"].nunique())
    print("test size:", result["test"]["movieId"].nunique())
    return result["train"], result["val"], result["test"]


def preprocess():
    labeled = load_labeled_dataset()
    ratings, movies = load_movielens()

    # returns train, val, test
    return split_relevant_judgements(labeled, ratings, movies)


if __name__ == '__main__':
    preprocess()
