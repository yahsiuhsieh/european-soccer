import torch
import pandas as pd

from models import Soccernet, Average


def get_match_stat(match, player_data):
    """
    Get FIFA stats for a given match
    """
    match_date = match["date"]
    players = [
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

    total_player_stat = pd.DataFrame(columns=["match_api_id"])

    # store match id
    total_player_stat.loc[0] = match["match_api_id"]

    # store each player's features
    for player in players:
        player_id = match[player]
        player_stat = player_data[player_data["player_api_id"] == player_id]

        # choose most recent player stat and interested features
        player_stat = player_stat[player_stat["date"] < match_date][0:1]
        player_stat = player_stat[player_stat.columns[4:]]

        # rename keys
        new_keys = []
        for key in player_stat.columns:
            key_name = "{}_{}".format(player, key)
            new_keys.append(key_name)
        player_stat.columns = new_keys
        player_stat.reset_index(inplace=True, drop=True)

        # concate data
        total_player_stat = pd.concat([total_player_stat, player_stat], axis=1)

    return total_player_stat.loc[0]


def get_FIFA(match_data, player_data, path="", load_data=False):
    """
    Get FIFA data for all matches
    """
    if load_data:
        print("Loading FIFA data...")
        fifa_data = pd.read_pickle(path + "fifa_data.pkl")
        print("Finish")
    else:
        print("Collecting FIFA data...")
        fifa_data = match_data.apply(lambda x: get_match_stat(x, player_data), axis=1)
        print("Finish")

        # Save data as pickle
        fifa_data.to_pickle("fifa_data.pkl", protocol=2)
    return fifa_data


def get_label_from_match(match):
    """
    Get label from a given match
    """
    label = pd.DataFrame()
    label.loc[0, "match_api_id"] = match["match_api_id"]
    home_goal = match["home_team_goal"]
    away_goal = match["away_team_goal"]

    if home_goal > away_goal:
        label.loc[0, "label"] = 0
    elif home_goal < away_goal:
        label.loc[0, "label"] = 1
    else:
        label.loc[0, "label"] = 2
    return label.loc[0]
