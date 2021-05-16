import numpy as np
import itertools
import pandas as pd
import collections
import json
import os
import sklearn
from sklearn.linear_model import ElasticNet
import re
import time
import lib
import prop_ranker as prop

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
    def grid_search(cls, implicit, supervised_train, supervised_val, metrics, verbose=False):
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
            val_scores = model.evaluate(supervised_val, metrics)
            results += [{"id": param_id, "params": params, "scores": val_scores}]

        # table = ScoreTable(results)
        # if verbose:
        #     print(table)
        return results

    def __init__(self, data, supervised_trainset=None):
        self._model = None
        self._my_settings = dict()
        self._data = data

    def score_candidates(self, item_id, candidate_ids=None, rank=False):
        if candidate_ids:
            candidate_ids = np.asarray(candidate_ids)
        else:
            candidate_ids = np.arange(self._data.n_items)

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
            candidate_ids, pred_score = self.score_candidates(ranking["a_iid"])
            for metric in metrics:
                scores[metric.name()] += [metric.score(ranking["labels"], pred_score)]
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
        self._norms = marginal.flatten()
        self._item_factors = self._ratings.T

    def _score_candidates(self, item_id, candidate_ids):
        return self.counts_ab(item_id, candidate_ids) / self.counts_b(item_id)

    def counts_ab(self, item_id, candidate_ids):
        item = self._item_factors[item_id]
        candidates = self._item_factors[candidate_ids]
        return np.asarray(candidates.dot(item.T).todense()).flatten()

    def counts_b(self, item_id):
        return self._norms[item_id]


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
        return self._data.get_popularity(candidate_ids).values


class SLIM(Recommender):
    # PARAM_GRID = {'l1_reg': [0.001, 0.01], 'l2_reg': [0.001, 0.01]}
    PARAM_GRID = {'l1_reg': [0.001], 'l2_reg': [0.01]}

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
        alpha = l1_reg + l2_reg
        l1_ratio = l1_reg / alpha

        self._model = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, positive=True, fit_intercept=False)

    def _score_candidates(self, item_id, candidate_ids):
        if item_id not in self._item_weights:
            self._fit()
            data = self._ratings[item_id]
            self._ratings[item_id] = 0
            self._model.fit(self._ratings.T, data.toarray().reshape(-1))
            self._ratings[item_id] = data
            self._item_weights[item_id] = self._model.coef_.copy()

        return self._item_weights[item_id][candidate_ids]


class ItemKNN(Cooccur):
    # PARAM_GRID = {'alpha': [0.3, 0.5, 0.7], 'lmbda': [0.0, 10.0, 20.0]}
    PARAM_GRID = {'alpha': [0.3], 'lmbda': [0.0]}

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

class DebiasedModel(Recommender):
    def __init__(self, data, labeled_rankings, model="ranker"):
        self._data = data
        self._cc = Cooccur(self._data)
        self._cc.fit()
        self._model = model
        self._labeled_rankings = labeled_rankings
        self._raw_features = ["popularity_log", "year", "year_first"]

    def _prepare_for_propensity_learning(self, rankings):
        cc = self._cc

        def augment(grp):
            b = grp.name
            a = grp["b_iid"].values
            grp["counts_ab"] = cc.counts_ab(b, a)
            grp["counts_b"] = cc.counts_b(b)

            grp["features"] = list(self._get_features(b, a))
            return grp

        rankings = rankings.groupby("a_iid").apply(augment)
        rankings["MLE_est"] = rankings["counts_ab"]/rankings["counts_b"]
        return rankings

    def _add_negative_samples(self, rankings, n_neg=500):
        def add_negative(grp):
            pos = grp[grp["label"] > 0].reset_index(drop=True)
            positive_ids = np.append(pos["b_iid"].values, [grp.name])

            # sample negative items at random
            negative_ids = np.setdiff1d(np.arange(self._data.n_items), positive_ids)
            neg_samples = np.random.choice(negative_ids, n_neg, replace=False)

            neg_df = pd.DataFrame({"a_iid": [grp.name]*len(neg_samples),
                                   "b_iid": neg_samples, "label": np.zeros_like(neg_samples)})
            df = pd.concat([pos, neg_df])
            return df

        return rankings.groupby("a_iid").apply(add_negative).reset_index(drop=True)

    def _learn_propensities(self):
        labeled_rankings = self._labeled_rankings["rows"]
        labeled_rankings = self._add_negative_samples(labeled_rankings)

        rankings = self._prepare_for_propensity_learning(labeled_rankings)
        cols = {"offset": "counts_ab", "features": "features", "y": "label"}

        X, y, offset = prop.RankingToPairs(column_mapping=cols).transform(rankings)
        print("Transformed to %d pairs" % len(y))
        ranker = prop.Ranker(model="linear", lr=0.02, n_steps=300)

        ranker.fit(X, y, offset)
        ranker.print_params()
        self._ranker = ranker

    def _fit(self):
        self._learn_propensities()

    def _get_features(self, item_id, candidate_ids):
        features = list()
        data = self._data
        for att in self._raw_features:
            features += [data.items.loc[candidate_ids, att].values]
            features += [abs(data.items.loc[candidate_ids, att].values -
                             data.items.loc[item_id, att])]

        cc = self._cc
        features += [cc.counts_ab(item_id, candidate_ids)/cc.counts_b(candidate_ids)]
        features = np.asarray(features).T
        return features

    def _score_candidates(self, item_id, candidate_ids):
        counts_ab = self._cc.counts_ab(item_id, candidate_ids)
        X = self._get_features(item_id, candidate_ids)
        return self._ranker.rank(X, counts_ab)
