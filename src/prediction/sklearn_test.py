import argparse
import sqlite3
from utils import *

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", type=str, default="false")
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = get_args()

    # get data
    database = "../database.sqlite"
    data = sqlite3.connect(database)
    matches = pd.read_sql("SELECT * FROM Match", data)
    player_attributes = pd.read_sql("SELECT * FROM Player_Attributes", data)

    # interested match columns
    cols = [
        "date",
        "match_api_id",
        "home_team_api_id",
        "away_team_api_id",
        "home_team_goal",
        "away_team_goal",
        "home_player_1",
        "home_player_2",
        "home_player_3",
        "home_player_4",
        "home_player_5",
        "home_player_6",
        "home_player_7",
        "home_player_8",
        "home_player_9",
        "home_player_10",
        "home_player_11",
        "away_player_1",
        "away_player_2",
        "away_player_3",
        "away_player_4",
        "away_player_5",
        "away_player_6",
        "away_player_7",
        "away_player_8",
        "away_player_9",
        "away_player_10",
        "away_player_11",
    ]

    # get FIFA data
    matches = matches.dropna(subset=cols)[cols]
    fifa_data = get_FIFA(matches, player_attributes, load_data=False)
    fifa_data = fifa_data.dropna().select_dtypes(["number"])

    # get label
    labels = matches.apply(get_label_from_match, axis=1)
    labels = labels[labels["match_api_id"].isin(fifa_data["match_api_id"])]

    # data preprocessing
    data = pd.merge(fifa_data, labels, on="match_api_id")
    data.drop("match_api_id", axis=1, inplace=True)

    # extract features and label data
    labels = data.loc[:, "label"]
    input_features = data.drop("label", axis=1)

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, labels, test_size=0.1, shuffle=True
    )

    # sklearn model testing
    GNB_clf = GaussianNB()  # Gaussian Naive Bayes
    KNN_clf = KNeighborsClassifier()  # K-Nearest Neighbors

    if args.pca == "false":
        GNB_clf.fit(X_train, y_train)
        print("GNB Success Rate: {:.2f} %".format(GNB_clf.score(X_test, y_test) * 100))

        KNN_clf.fit(X_train, y_train)
        print("KNN Success Rate: {:.2f} %".format(KNN_clf.score(X_test, y_test) * 100))

    elif args.pca == "true":
        pca = PCA(n_components=5)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        GNB_clf.fit(X_train_pca, y_train)
        print(
            "GNB Success Rate: {:.2f} %".format(GNB_clf.score(X_test_pca, y_test) * 100)
        )

        KNN_clf.fit(X_train_pca, y_train)
        print(
            "KNN Success Rate: {:.2f} %".format(KNN_clf.score(X_test_pca, y_test) * 100)
        )
