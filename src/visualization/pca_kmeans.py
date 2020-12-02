import sqlite3
from visual_utils import *

if __name__ == "__main__":
    database = "../database.sqlite"
    data = sqlite3.connect(database)

    player_attributes = pd.read_sql("SELECT * FROM Player_Attributes", data)
    player_attributes = player_attributes.dropna()

    # select features used by k-means model
    select_features = [
        "overall_rating",
        "potential",
        "crossing",
        "finishing",
        "heading_accuracy",
        "short_passing",
        "volleys",
        "dribbling",
        "curve",
        "free_kick_accuracy",
        "long_passing",
        "ball_control",
        "acceleration",
        "sprint_speed",
        "agility",
        "reactions",
        "balance",
        "shot_power",
        "jumping",
        "stamina",
        "strength",
        "long_shots",
        "aggression",
        "interceptions",
        "positioning",
        "vision",
        "penalties",
        "marking",
        "standing_tackle",
        "sliding_tackle",
        "gk_diving",
        "gk_handling",
        "gk_kicking",
        "gk_positioning",
        "gk_reflexes",
    ]
    df_select = player_attributes[select_features]
    num_features = len(select_features)

    # get k-means model for player dataset
    model = k_means_model(df_select, numOfClusters=3)

    # Figure 1: plot out comparison plot for three class representativs
    select_PCA_features = classRep_comparisonPlot(model, select_features)

    # Figure 2: plot out 3-component pca result
    principalDf = PCA_vis(select_PCA_features, player_attributes)

    # get k-means model for pca result
    model_kmeans_pca = k_means_model(principalDf, numOfClusters=3)

    # Figure 3: plot out comparison plot for three class representativs
    selectfeatures_pca = [
        "principal component 1",
        "principal component 2",
        "principal component 3",
    ]
    classRep_comparisonPlot_pca(model_kmeans_pca, selectfeatures_pca)

    # Figure 4: plot ke-means result on pca
    kmeans_vis(principalDf)
