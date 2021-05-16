import scipy.sparse as sp
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import collections
import json

PATH = "datasets/"


def from_json(file):
    with open(file, encoding='utf-8') as fh:
        return json.load(fh)


def to_json(file, data):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w', encoding='utf8') as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def ml(x, fpath=PATH):
    f_tags = fpath + "ml-25m/genome-scores.csv"
    prefix = fpath + x + "/"

    return (prefix + "ratings.csv", prefix + "movies.csv", f_tags)


def read_movielens_ratings(movielens_file):
    params = {"skiprows": 1, "engine": "c", "header": None}
    ratings_raw = pd.read_csv(movielens_file, names=[
        "userId", "movieId", "rating", "timestamp"], **params)
    print("I have read %d lines " % len(ratings_raw))

    i = ratings_raw["movieId"].values
    j = ratings_raw["userId"].values

    ratings = sp.coo_matrix((ratings_raw["rating"].values, (i, j))).transpose()
    timestamps = sp.coo_matrix((ratings_raw["timestamp"].values, (i, j))).transpose()

    return ratings, timestamps


class MovieLens(object):
    def __init__(self):
        movies = pd.read_csv("data/processed/movies_1.csv", index_col="newId")
        ratings = pd.read_csv("data/processed/ratings_coo_1.csv", names=["userId", "movieId", "rating"], header=None)

        i = ratings["userId"].values
        j = ratings["movieId"].values
        data = (ratings["rating"]).astype(int)

        ratings = sp.coo_matrix((data, (i, j))).tocsc()


        newId_lut = -np.ones(movies.movieId.max() + 2, dtype=int)
        newId_lut[movies.movieId.values] = movies.index.values

        popularity = np.asarray(ratings.sum(axis=0)).flatten()
        index = np.arange(len(popularity), dtype=int)
        movies.loc[index, "popularity"] = popularity
        movies["normalized_popularity"] = (
                movies["popularity"] / (movies["year"].max() + 1 - movies["year"]))
        movies["popularity_rank"] = movies["popularity"].rank(method="first", ascending=False)
        movies["popularity_log"] = np.log(movies["popularity"])
        self.ratings = ratings
        self.newIdLookupTable = newId_lut
        self._tags = "NOT_INITIALIZED"
        self.items = movies
        self.ratings_file = movies
        self.f_tags = movies
        self.n_items = ratings.shape[1]
        self.n_users = ratings.shape[0]

        # popularity = ratings[["movieId", "rating"]].groupby("movieId").sum().reset_index()
        # popularity.rename(columns={"rating": "popularity"}, inplace=True)
        # popularity["log_popularity"] = np.log(popularity["popularity"])
        # popularity["popularity_rank"] = popularity["popularity"].rank(method="first", ascending=False)
        #
        # movies = pd.merge(movies, popularity, left_on="movieId", right_on="movieId")
        #
        # # movies["normalized_popularity"] = movies["popularity"]/movies["year"].max() + 1 - movies["year"]
        #
        # self.movies = movies
        # self.n_movies = movies["movieId"].nunique()
        # self.movie_list = np.asarray(movies["movieId"].unique())
        # self.ratings = ratings
        #
        # # self.ratings_converted = ratings.groupby("userId")["movieId"].apply(list).reset_index()
        # # self.ratings_converted.columns = ["userId", "movieIds"]
        #
        # self.ratings_converted_2 = ratings.groupby("movieId")["userId"].apply(list).reset_index()
        # self.ratings_converted_2.columns = ["movieId", "userIds"]

    def get_popularity(self, movie_id):
        # return np.asarray(self.movies[self.movies.movieId == movie_id]["popularity"]).astype(np.float32)
        return self.items.loc[movie_id, "popularity"].apply(float)

    def ml_id_to_internal_id(self, ml_ids):
        return np.take(self.newIdLookupTable, ml_ids, mode='clip')

    def load_dataset(self, file_name):
        # dataset = pd.read_csv(file_name)
        #
        # true_labels = pd.DataFrame(columns=["movieId", "labels"])
        # for name, group in dataset.groupby("movieId"):
        #     labels = group[group["sim_bin"] == 1]["neighborId"].tolist()
        #
        #     true_labels = true_labels.append({"movieId": name, "labels": labels}, ignore_index=True)
        #
        # return {"groups": true_labels}

        rankings = pd.read_csv(file_name)

        rankings["a_iid"] = self.ml_id_to_internal_id(rankings["movieId"].values)
        rankings["b_iid"] = self.ml_id_to_internal_id(rankings["neighborId"].values)

        valid_rankings = rankings[(rankings.a_iid > -1) & (rankings.b_iid > -1)]

        info = collections.OrderedDict()
        info["# rankings"] = rankings["movieId"].nunique()
        info["# invalid rankings"] = (rankings.groupby(by="movieId").max()["a_iid"] == -1).sum()
        info["# invalid results"] = (rankings["b_iid"] == -1).sum()
        per_ranking = valid_rankings.groupby(by="a_iid").count()["b_iid"]
        labels_per_ranking = valid_rankings.groupby(by="a_iid").sum()["sim_bin"]
        info["Length of each ranking"] = ("%.2f (min=%d, max=%d))" % (
            per_ranking.mean(), per_ranking.min(), per_ranking.max()))
        info["Pos labels in each ranking"] = ("%.2f (min=%d, max=%d))" % (
            labels_per_ranking.mean(), labels_per_ranking.min(), labels_per_ranking.max()))

        print(info)

        df = pd.DataFrame({"a_iid": valid_rankings["a_iid"], "a_ml_id": valid_rankings["movieId"],
                           "b_iid": valid_rankings["b_iid"], "b_ml_id": valid_rankings["neighborId"],
                           "label": valid_rankings["sim_bin"]})

        As, Bs = list(), list()
        for name, group in df.groupby("a_iid"):
            labels = np.zeros(self.n_items)
            labels[group["b_iid"].values.flatten()] = group["label"].values.flatten()
            As.append(name)
            Bs.append(labels)
        groups = pd.DataFrame({"a_iid": As, "labels": Bs})

        return {"rows": df, "groups": groups}

def read_movie_titles(file):
    if "100k" in file:
        df = pd.read_csv(file, sep="|", engine="python", usecols=[
            0, 1], header=None, names=["movieId", "title"])
    elif "1m" in file or "10m" in file:
        df = pd.read_csv(file, sep="::", engine="python", usecols=[
            0, 1, 2], header=None, names=["movieId", "title", "genres"])
    else:
        df = pd.read_csv(file)
    return df


def extract_timestamp_features(timestamps, remaining_movie_ids):
    print("Converting timestamps...")
    timestamps = timestamps.T.tolil()
    t_index = pd.date_range(start="1996", end="2020", freq='A-DEC')

    results = list()
    for i, mid in enumerate(remaining_movie_ids):
        df = pd.DataFrame({"time": pd.to_datetime(
            timestamps.data[mid], unit="s")}).set_index('time')
        df["event"] = 1
        ts = df.resample('Y').count().reindex(t_index).fillna(0)
        skew = ts.event.skew()
        peak = ts.event.idxmax().year
        coeffvar = ts["event"].std() / ts["event"].mean()
        first = ts.event.ne(0).idxmax().year
        results += [{"movieId": mid, "time_peak": peak,
                     "time_first": first, "time_skew": skew, "tm_var": coeffvar}]

        if (i % 50) == 0:
            print("Progress:  %d / %d" % (i, len(remaining_movie_ids)))

    return pd.DataFrame(results).set_index("movieId")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("movielens_version", type=str, choices=["100k", "20m", "25m"],
                        help="increase output verbosity")
    parser.add_argument("--min_rating", type=float, default=3.0,
                        help="ratings >= min_rating will be considered positive")
    parser.add_argument("--min_count", type=int, default=30,
                        help="movies with < min_count positive ratings will be removed")
    args = parser.parse_args()
    # convert_to_implicit(args.movielens_version, args.min_rating, args.min_count)
