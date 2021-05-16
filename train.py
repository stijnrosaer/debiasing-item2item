import lib
from lib import MovieLens
from recommenders import *
from preprocess import preprocess

# METHODS = [Random, Popularity, Cooccur, SLIM, ItemKNN]
METHODS = [Random, Popularity, Cooccur, SLIM, ItemKNN, DebiasedModel]
METHODS = [DebiasedModel]


OUTFILE = "results/2a_find_best.json"

dataset = MovieLens()
metric = lib.metrics.RecallAtK(100)
metrics = [lib.metrics.MeanRanks(), lib.metrics.RecallAtK(
    100), lib.metrics.RecallAtK(50), lib.metrics.RecallAtK(25), lib.metrics.PrecisionAtK(100)]


train = dataset.load_dataset("data/labeled/train.csv")
test = dataset.load_dataset("data/labeled/test.csv")
val = dataset.load_dataset("data/labeled/val.csv")
# train, val, test = preprocess()

p = []
for i in train["groups"]["a_iid"]:
    p.append(i)
print("train:", np.median(p))

p = []
for i in test["groups"]["a_iid"]:
    p.append(i)
print("test:", np.median(p))

p = []
for i in val["groups"]["a_iid"]:
    p.append(i)
print("val:", np.median(p))

results = dict()
for model in METHODS:
    print("\n\nCurrent model " + model.__name__)
    result = model.grid_search(dataset, train,
                               val, metrics, verbose=True)

    results[model.__name__] = result
    lib.to_json(OUTFILE, results)

# test: 4410 --> 0.683654
# val: 5017 --> 0.764191