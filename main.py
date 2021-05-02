from preprocess import preprocess
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    train, val, test = preprocess()
    clf = LogisticRegression()
    clf.fit(train["movieId", "neighborId"], train["sim_bin"])
    print(clf.score(test["movieId", "neighborId"], test["sim_bin"]))


