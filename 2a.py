import lib
from lib import MovieLens
from reccomenders import *
from preprocess import preprocess

METHODS = [Random, ItemKNN, Cooccur, Popularity, SLIM]
OUTFILE = "results/2a_find_best.json"

dataset = MovieLens()
metric = lib.metrics.RecallAtK(100)

train, val, test = preprocess()

results = dict()
for model in METHODS:
    print("\n\nCurrent model " + model.__name__)
    result = model.grid_search(dataset, train,
                               test, metric, verbose=True)

    results[model.__name__] = result.to_dict()
    lib.to_json(OUTFILE, results)