import sqlite3
from visual_utils import *

if __name__ == "__main__":
    database = "../database.sqlite"
    data = sqlite3.connect(database)

    leagues = pd.read_sql(
        "SELECT * FROM League JOIN Country on Country.id == League.id", data
    )
    teams = pd.read_sql("SELECT * FROM Team", data)
    players = pd.read_sql("SELECT * FROM Player", data)
    player_attributes = pd.read_sql("SELECT * FROM Player_Attributes", data)
    matches = pd.read_sql("SELECT * FROM Match", data)

    # plots
    build_pie_plot(matches)
    build_player_atrribute_corelation(player_attributes)
    build_sorted_corelation(player_attributes)
    build_preferred_foot_and_potential_distribution(player_attributes)
    build_home_advantage(matches)
    build_radar_chart(players, player_attributes, num_of_top_players=10)
