import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.pipeline

import sklearn.neighbors
import torch.nn.utils.rnn

np.random.seed(42)
torch.manual_seed(42)


def no_warn_log(x):
    result = -np.ones_like(x) * np.inf
    result[x > 0] = np.log(x[x > 0])
    return result

def no_warn_log_2(x):
    result = -np.ones_like(x) * np.inf
    result[x > 0] = x[x > 0]
    return result


class Ranker(object):
    models = {"linear": lambda dim: torch.nn.Linear(dim, 1, bias=False),
              "2layer": lambda dim: torch.nn.Sequential(
                  torch.nn.Linear(dim, 50),
                  torch.nn.LeakyReLU(),
                  torch.nn.Linear(50, 1),
              ),
              "3layer": lambda dim: torch.nn.Sequential(
                  torch.nn.Linear(dim, 40),
                  torch.nn.LeakyReLU(),
                  torch.nn.Linear(40, 5),
                  torch.nn.LeakyReLU(),
                  torch.nn.Linear(5, 1),
              )
              }

    def __init__(self, lr=0.5, n_steps=1000, verbose=True, model="linear"):
        self._lr = lr
        self._n_steps = n_steps
        self._params = None
        self._verbose = verbose
        poly = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(
        ), sklearn.preprocessing.PolynomialFeatures(2), sklearn.preprocessing.StandardScaler())
        self._scaler = sklearn.pipeline.FeatureUnion([("poly", poly)])
        self._scaler = sklearn.preprocessing.StandardScaler()
        self._model = self.models[model]

    def fit(self, X, y, offset):
        X_i = torch.tensor(self._scaler.fit_transform(X[0]), dtype=torch.double)
        X_j = torch.tensor(self._scaler.transform(X[1]))

        offset = torch.tensor(offset.reshape(-1, 1), dtype=torch.double)
        model = self._model(X_i.size(1)).double()

        if self._verbose:
            print("Length of input", X_i.shape)

        def pred():
            output = model(X_i) - model(X_j)
            return output + offset

        def loss():
            loss = F.relu(-pred())
            loss = loss.mean()
            _optimizer.zero_grad()
            loss.backward()
            return loss

        def percentage_violated():
            return (pred() < 0).float().sum().item()

        _optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        if self._verbose:
            all = pred()
            invalid = all[~torch.isfinite(all)]
            if len(invalid):
                print("Warning: Found ", len(invalid), "invalid pairs")
            print("\tloss\t#constraints violated")
        for step in range(self._n_steps):
            current_loss = _optimizer.step(loss)
            if (step % 50 == 0 or step == self._n_steps - 1) and self._verbose:
                print("", "%.2e" % current_loss.item(), "%.2f" % percentage_violated(), sep="\t")
        self._model = model

        self._params = {n: p.data.cpu().numpy().T for n, p in model.named_parameters()}

    def print_params(self):
        print(self._params)

    def rank(self, X, counts_ab):
        p_inv = self.transform(X, exp=False).flatten()
        return no_warn_log(counts_ab.flatten()) + p_inv
        # return mle.flatten() + p_inv

    def transform(self, X, exp=True):
        X = self._scaler.transform(X)
        scores = self._model(torch.as_tensor(X)).data.cpu().numpy().flatten()
        if exp:
            return np.exp(scores)
        else:
            return scores


class RankingToPairs(object):
    def __init__(self, column_mapping=dict()):
        default = {"q": "a_iid", "y": "sim_ab", "counts_b": "counts_b", "offset": "MLE_est",
                   "counts_ab": "counts_ab", "features": "features"}
        self._column_mapping = {**default, **column_mapping}

    @staticmethod
    def _induce_pairs(indices, vals):
        n = len(vals)
        b, a = np.meshgrid(np.arange(n), np.arange(n))
        valid = (vals[a] > vals[b])
        return indices[a[valid]], indices[b[valid]]

    @staticmethod
    def induce_pairs(y, q):
        df = pd.DataFrame({"q": q, "y": y})
        pairs = list()
        for q_id, grp in df.groupby("q"):
            t1, t2 = RankingToPairs._induce_pairs(grp.index.values, grp.y.values)
            _pairs = list(zip(t1.tolist(), t2.tolist()))
            pairs += _pairs
        pairs = np.asarray(pairs, dtype=int)
        return pairs[:, 0], pairs[:, 1]

    def transform(self, df):
        cols = self._column_mapping
        X = np.asarray(df[cols["features"]].tolist())
        q = df[cols["q"]].values
        y = df[cols["y"]].values
        offsets = no_warn_log(df[cols["offset"]].values)

        i, j = self.induce_pairs(y, q)
        X_new = (np.take(X, i, axis=0), np.take(X, j, axis=0))
        y_new = np.ones_like(i)
        offsets_new = np.take(offsets, i, axis=0) - np.take(offsets, j, axis=0)

        filtered = np.isfinite(offsets_new)

        return (X_new[0][filtered], X_new[1][filtered]), y_new[filtered], offsets_new[filtered]
