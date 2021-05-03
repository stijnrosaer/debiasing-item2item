import numpy as np
import itertools
import pandas as pd
import collections
import json
import os
import sklearn
import re
import time

MODEL_CACHE = "data/cache"


class ScoreTable(object):
    def __init__(self, results):
        self._df = pd.DataFrame(results)
        self._argmax = self._df.loc[self._df["score"].idxmax()]

    def best_param(self):
        return self._argmax["params"]

    def best_score(self):
        return self._argmax["score"]

    def __str__(self):
        return self._df.to_string()

    def to_dict(self):
        return {"best_param": self.best_param(), "best_score": self.best_score(),
                "all_scores": self._df.to_dict('records')}

class Cache(object):
    def __init__(self, instance):
        class_name = instance.__class__.__name__
        self._instance = instance
        self.base_path = MODEL_CACHE + "/" + class_name + "/"
        self.model_index = self.base_path + "available_models.json"
        self._loaded_file = None
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        if os.path.isfile(self.model_index):
            with open(self.model_index, encoding='utf-8') as fh:
                self._models = json.load(fh)
        else:
            self._models = dict()

    def load(self):
        id = self._instance.get_id()
        if id in self._models:
            modelfile = self._models[id]
            modelfile = self.base_path + modelfile
            if os.path.isfile(modelfile):
                self._loaded_file = self._models[id]
                return np.load(modelfile)
            else:
                del self._models[id]
                self.write_index()
        else:
            return None

    def store(self):
        id = self._instance.get_id()
        if not self._loaded_file:
            filename = re.sub("[^0-9a-zA-Z=.,]+", "", id) + "_" + str(int(time.time())) + ".npz"
            self._models[id] = filename
        else:
            filename = self._loaded_file

        np.savez(self.base_path + filename, **self._instance.get_params(), id=self._instance.get_id())
        self._loaded_file = filename
        self.write_index()

    def write_index(self):
        with open(self.model_index, 'w', encoding='utf8') as json_file:
            json.dump(self._models, json_file, ensure_ascii=False)

    def rebuild_index(self):
        new_index = dict()
        for file in os.listdir(self.base_path):
            if file.endswith(".npz"):
                id = np.load(self.base_path + file)["id"].tolist()
                if id in new_index:
                    print("Warning - Following id already exists: ", id)
                new_index[id] = file
        self._models = new_index
        self.write_index()

class Recommender(object):
    PARAM_GRID = dict()

    @classmethod
    def grid_search(cls, implicit, supervised_train, supervised_val, metric, verbose=False):
        def product_dict(kwargs):
            keys = kwargs.keys()
            vals = kwargs.values()
            for instance in itertools.product(*vals):
                yield dict(zip(keys, instance))

        results = list()
        for param_id, params in enumerate(product_dict(cls.PARAM_GRID)):
            if verbose:
                print("Fitting ", params)
            model = cls(implicit, supervised_train, **params)
            model.fit()
            val_score = list(model.evaluate(supervised_val, [metric]).values())[0]["mean"]
            results += [{"id": param_id, "params": params, "score": val_score}]

        table = ScoreTable(results)
        if verbose:
            print(table)
        return table

    def __init__(self, data, supervised_trainset=None):
        self._model = None
        self._my_settings = dict()
        self._data = data

    def score_candidates(self, item_id, candidate_ids=None, rank=False):
        if candidate_ids:
            candidate_ids = np.asarray(candidate_ids)
        else:
            candidate_ids = self._data.movie_list
        scores = self._score_candidates(item_id, candidate_ids)
        scores[candidate_ids == item_id] = -np.inf  # exclude item itself
        if rank:
            argsort = np.argsort(-scores)
            scores_sorted = scores[argsort]
            candidates_sorted = candidate_ids[argsort]
            return candidates_sorted, scores_sorted
        else:
            return candidate_ids, scores

    def evaluate(self, rankings, metrics):
        rankings = rankings["groups"]
        if not len(rankings):
            raise ValueError("Length of rankings must be at least one!")
        scores = collections.defaultdict(list)
        for _, ranking in rankings.iterrows():
            candidate_ids, pred_score = self.score_candidates(ranking["movieId"])
            for metric in metrics:
                scores[metric.name()] += [metric.score(ranking["labels"], pred_score, candidate_ids)]
        return {k: {"mean": np.mean(v), "raw": v} for k, v in scores.items()}

    def fit(self):
        self._fit()


class Cooccur(Recommender):
    def __init__(self, data, supervised_trainset=None):
        super().__init__(data)
        self._ratings = data.ratings
        self._data = data

    def _fit(self):
        marginal = np.asarray(self._ratings.sum(axis=0))
        marginal[marginal == 0] = 1  # prevent division by 0

    def _score_candidates(self, item_id, candidate_ids):
        return self.counts_ab(item_id, candidate_ids) / self.counts_b(item_id)

    def counts_ab(self, item_id, candidate_ids):
        print(candidate_ids)
        print(item_id)
        print(self._ratings.head())
        print(self._data.ratings_converted.head())

        # item2 = self._data.ratings_converted_2[self._data.ratings_converted_2["movieId"] == item_id]
        # item_2_list = item2["userIds"].tolist()
        # candidates = self._data.ratings_converted_2[self._data.ratings_converted_2["movieId"] == candidate_ids]
        # print(candidates.head())
        #
        # return np.asarray(candidates.dot(item2.T).todense()).flatten()

        mask = self._data.ratings_converted.movieIds.apply(lambda x: item_id in x)
        tmp = self._data.ratings_converted[mask]
        # item = self._data.ratings_converted[self._data.ratings_converted["movieIds"].contains(item_id)]
        a = pd.Series([item for sublist in tmp.movieIds for item in sublist])
        print(a)
        counts = a.groupby(a).size().rename_axis('count').reset_index(name='movieId')
        print(counts.head())
        print(len(counts))
        print(counts["movieId"].nunique())
        rval = []
        for i in candidate_ids:
            rval.append(counts[counts["movieId"] == i]["count"].values)
        print(rval[:5])
        return rval
        # item = self._item_factors[item_id]
        # candidates = self._item_factors[candidate_ids]
        # return np.asarray(candidates.dot(item.T).todense()).flatten()

    def counts_b(self, item_id):
        sessions = len(self._ratings[self._ratings["movieId"] == item_id].groupby("userId"))
        return sessions



class CachedRecommender(Recommender):
    def __init__(self, data, supervised_trainset=None):
        super().__init__(data)
        self._cache = Cache(self)

    def fit(self):
        cached_params = self._cache.load()
        if cached_params is None:
            print("Couldn't find saved model. Training.")
            self._fit()
            self._cache.store()
        else:
            print("Found cached model.")
            cached_params = {k: v for k, v in cached_params.items() if k != "id"}
            self.set_params(cached_params)


class Random(Recommender):
    def __init__(self, data, supervised_trainset=None):
        self._data = data

    def _fit(self):
        pass

    def _score_candidates(self, movie_id, candidate_ids):
        rng = np.random.RandomState(movie_id)
        return rng.rand(len(candidate_ids))

class Popularity(Recommender):
    def _fit(self):
        pass

    def _score_candidates(self, movie_id, candidate_ids):
        return self._data.get_popularity(candidate_ids)


class SLIM(CachedRecommender):
    PARAM_GRID = {'l1_reg': [0.001, 0.01], 'l2_reg': [0.001, 0.01]}

    def __init__(self, data, supervised_data=None, l1_reg=0.001, l2_reg=0.01):
        super().__init__(data)
        self._my_settings["l1_reg"] = l1_reg
        self._my_settings["l2_reg"] = l2_reg

        self._ratings = data.ratings.T.tolil()
        self._data = data
        self._item_weights = dict()

    def _fit(self):
        l1_reg = self._my_settings["l1_reg"]
        l2_reg = self._my_settings["l2_reg"]
        alpha = l1_reg+l2_reg
        l1_ratio = l1_reg/alpha

        self._model = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False)

    def get_params(self):
        return {str(k): v for k, v in self._item_weights.items()}

    def set_params(self, cached_params):
        self._item_weights = {int(k): v for k, v in cached_params.items()}

    def _score_candidates(self, item_id, candidate_ids):
        if item_id not in self._item_weights:
            self._fit()
            data = self._ratings[item_id]
            self._ratings[item_id] = 0
            self._model.fit(self._ratings.T, data.toarray().reshape(-1))
            self._ratings[item_id] = data
            self._item_weights[item_id] = self._model.coef_.copy()
            self._cache.store()

        return self._item_weights[item_id][candidate_ids]

class ItemKNN(Cooccur):
    PARAM_GRID = {'alpha': [0.3, 0.5, 0.7], 'lmbda': [0.0, 10.0, 20.0]}

    def __init__(self, data, supervised_data=None, alpha=0.5, lmbda=0.0):
        super().__init__(data)
        self._my_settings["alpha"] = alpha
        self._my_settings["lmbda"] = lmbda

    def _fit(self):
        self._norms = np.asarray(self._ratings.sum(axis=0)).flatten()
        self._item_factors = self._ratings.T

    def _score_candidates(self, item_id, candidate_ids):
        alpha, lmbda = self._my_settings["alpha"], self._my_settings["lmbda"]
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        joint = np.asarray(candidates.dot(item.T).todense())

        candidate_norms = np.power(self._norms[candidate_ids] + lmbda, 1.0 - alpha)
        norms = candidate_norms

        item_norm = np.power(self._norms[item_id] + lmbda, alpha)
        norms *= item_norm

        norms[norms == 0] = 1

        return joint.flatten() / norms
