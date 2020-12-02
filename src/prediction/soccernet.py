import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from models import *
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # get data
    database = "../database.sqlite"
    data = sqlite3.connect(database)

    leagues = pd.read_sql(
        "SELECT * FROM League JOIN Country on Country.id == League.id", data
    )
    teams = pd.read_sql("SELECT * FROM Team", data)
    player_attributes = pd.read_sql("SELECT * FROM Player_Attributes", data)
    matches = pd.read_sql("SELECT * FROM Match", data)

    # get interested match columns
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
    fifa_data = get_FIFA(matches, player_attributes, load_data=True)
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

    # get training and testing data from DataLoader
    X_train = X_train.values.astype(np.float32)
    y_train = y_train.values.astype(np.int64)
    X_test = X_test.values.astype(np.float32)
    y_test = y_test.values.astype(np.int64)
    X_train, y_train, X_test, y_test = map(
        torch.tensor, (X_train, y_train, X_test, y_test)
    )

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=100, shuffle=True)

    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=100)

    # train and validate neural network
    model = Soccernet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(50):
        lr_step = [10, 20, 30, 40, 50]
        if epoch in lr_step:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1
        model.train(train_dl, criterion, optimizer, epoch)
        model.validate(test_dl, criterion)

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(model.train_acc)
    plt.plot(model.test_acc)
    plt.legend(("training", "validation"))
    plt.title("Accuracy v.s. Epochs")
    plt.xlabel("epochs")
    plt.ylabel("accuracy (%)")
    plt.show()