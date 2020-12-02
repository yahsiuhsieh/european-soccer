import math
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib.pyplot as plt

from itertools import cycle, islice
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans


def build_pie_plot(matches):
    """
    Build the pie plot to show the distribution of the matches
    """
    pie_match_count = matches["league_id"].value_counts()
    list_label = [
        "Spain LIGA BBVA",
        "France Ligue 1",
        "England Premier League",
        "Italy Serie A",
        "Netherlands Eredivisie",
        "Germany 1. Bundesliga",
        "Portugal Liga ZON Sagres",
        "Poland Ekstraklasa",
        "Scotland Premier League",
        "Belgium Jupiler League",
        "Switzerland Super League",
    ]
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    fig1.set_size_inches(w=15, h=10)

    # emphasize on top 3 leauges
    pie_plot = plt.pie(
        pie_match_count,
        labels=list_label,
        shadow=True,
        explode=(0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0),
        autopct="%1.1F%%",
    )
    plt.show()


def build_player_atrribute_corelation(player_attributes):
    """
    Build the color map of player atrribute corelation
    """
    player_attributes_wo_na = player_attributes.dropna()
    player_attributes_corr = player_attributes_wo_na.corr()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=24, h=24)
    sns.heatmap(player_attributes_corr, annot=True, linewidths=0.5, ax=ax, cmap="Blues")
    plt.show()


def build_sorted_corelation(player_attributes):
    """
    Build the bar graph of positive player atrribute corelation
    """
    player_attributes_wo_na = player_attributes.dropna()
    player_attributes_corr = player_attributes_wo_na.corr()
    df_overall_rating_corr = player_attributes_corr["overall_rating"]

    df_single_corr = pd.DataFrame(df_overall_rating_corr)
    df_single_corr = df_single_corr.sort_values(by=["overall_rating"], ascending=True)

    # drop first three rows since we only care about positive correlation
    index_headers = list(df_single_corr[3:].index)
    plt.figure(figsize=(20, 10))
    plt.barh(
        y=index_headers,
        width=df_single_corr.overall_rating[3:],
        height=1,
        linewidth=0.5,
    )
    plt.show()


def build_preferred_foot_and_potential_distribution(player_attributes):
    """
    Build the bargraph of preferredfoot and the potential distribution
    """
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(20, 10)

    sns.countplot(x=player_attributes["preferred_foot"], ax=ax[0])
    sns.countplot(x=player_attributes["potential"], ax=ax[1])

    pl.xticks(rotation=270)
    fig.tight_layout()
    plt.show()


def build_home_advantage(matches):
    """
    Build the line chart of avg home team goal and avg away team goal to see whether there is home advantage or not
    """
    D = {}
    df_goal = matches[["season", "home_team_goal", "away_team_goal"]]

    for i in range(len(df_goal)):
        key_season = df_goal.iloc[i].season
        if key_season not in D:
            D[key_season] = [
                1,
                df_goal.iloc[i].home_team_goal,
                df_goal.iloc[i].away_team_goal,
            ]
        else:
            D[key_season][0] += 1
            D[key_season][1] += df_goal.iloc[i].home_team_goal
            D[key_season][2] += df_goal.iloc[i].away_team_goal

    for key in D:
        D[key][1] /= D[key][0]
        D[key][2] /= D[key][0]

    df_goal_info = pd.DataFrame(D)
    column_headers = list(df_goal_info.columns.values)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=7, h=4)
    plt.plot(column_headers, df_goal_info.iloc[1], label="avg_home_goal")
    plt.plot(column_headers, df_goal_info.iloc[2], label="avg_away_goal")
    pl.xticks(rotation=270)
    plt.xlabel("Season")
    plt.ylabel("Average Goal")
    plt.legend()
    plt.show()


def build_radar_chart(players, player_attributes, num_of_top_players=10):
    """
    Build the radar chart of top N players to see what attributes they have in common
    """

    df_overall_rating = player_attributes[["id", "player_api_id", "overall_rating"]]
    df_record = df_overall_rating.sort_values(by="overall_rating", ascending=False)
    df_record.drop_duplicates("player_api_id", "first", inplace=True)
    top_n_records = df_record[:num_of_top_players]
    index_headers = list(top_n_records.index)

    df_name = players[["id", "player_api_id", "player_name"]]
    top_n_players = pd.DataFrame()
    for i in range(len(top_n_records)):
        x = top_n_records.player_api_id.iat[i]
        top_n_players.loc[i, "Name"] = df_name[
            players.player_api_id == x
        ].player_name.iat[0]

    # choose interested features
    radar_info = pd.DataFrame()
    radar_info = player_attributes.iloc[index_headers, :]
    df = radar_info[
        [
            "overall_rating",
            "potential",
            "crossing",
            "long_passing",
            "ball_control",
            "reactions",
            "long_shots",
            "standing_tackle",
        ]
    ]

    categories = list(df)[:]
    N = len(categories)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(w=10, h=10)

    # plot
    for i in range(num_of_top_players):
        values = df.iloc[i].values.flatten().tolist()
        values += values[:1]

        # the angle of each axis in the plot
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color="grey", size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle="solid")

        # Fill area
        ax.fill(angles, values, "b", alpha=0.1)
    plt.show()


def k_means_model(df, numOfClusters):
    """
    Takes the dataset, num of clusters as input
    and returns the k-means model
    """
    # Perform scaling on the dataframe containing the selected features
    data = scale(df)

    # Train a model
    model = KMeans(init="k-means++", n_clusters=numOfClusters, n_init=20).fit(data)
    return model


def pd_centers(featuresUsed, centers):
    """
    Finding class representatives
    """
    colNames = list(featuresUsed)
    colNames.append("prediction")
    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P["prediction"] = P["prediction"].astype(int)
    return P


def parallel_plot(data, rg):
    """
    Plot the parallel plots for k-means class representatives
    """
    my_colors = list(islice(cycle(["b", "r", "g", "y", "k"]), None, len(data)))
    plt.figure(figsize=(18, 8)).gca().axes.set_ylim(rg)
    parallel_coordinates(data, "prediction", color=my_colors, marker="o")


def classRep_comparisonPlot(model, selectfeatures):
    """
    Takes select features and k-menas model as input,
    and it plot out the class representative comparison plot，
    it returns the feature list of features with large variance
    """
    num_features = len(selectfeatures)
    Pan = pd_centers(featuresUsed=selectfeatures, centers=model.cluster_centers_)
    Pan["prediction"] = Pan["prediction"].replace(
        [0, 1, 2], ["class1", "class2", "class3"]
    )

    # replace attribute names with numbers to avoid overlappings in parallel plot x-axis
    newColNames = list(range(num_features))
    res = {
        list(Pan.columns)[i]: newColNames[i] for i in range(len(list(Pan.columns)[:-1]))
    }
    Pan.rename(columns=res, inplace=True)

    # parallel plots for three class representatives
    parallel_plot(Pan, [-3, +3.5])
    plt.xlabel("Attibutes Indices", fontsize=15)
    plt.ylabel("Attibute values", fontsize=15)
    plt.title("Class Representatives Comparison", fontsize=20)
    plt.show()

    # featureNum contains features that have relatively larger variances for three classes
    features_with_big_var = [
        0,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        21,
        24,
        25,
        26,
        27,
        28,
        29,
    ]
    features_lst = []
    for fName, fNum in res.items():
        if fNum in features_with_big_var:
            features_lst.append(fName)
    return features_lst


def classRep_comparisonPlot_pca(model_kmeans_pca, selectfeatures_pca):
    """
    Takes select features and k-menas model as input,
    and plot out the class representative comparison plot for PCA results
    """
    Pan_pca = pd_centers(
        featuresUsed=selectfeatures_pca, centers=model_kmeans_pca.cluster_centers_
    )
    # parallel plot for three class representatives of K-means on PCA result
    parallel_plot(Pan_pca, [-0.9, 2.3])
    return None


def PCA_vis(select_PCA_features, player_attributes):
    """
    Takes feature list for features selected for PCA analysis
    and player dataframe as input, it plot out the 3-component PCA result and
    returns the dataframe of player dataset's projection onto three principal
    components
    """
    x = player_attributes.loc[:, select_PCA_features].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # perform 3 component PCA
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    # plot players dataset projection on three principal components
    #     %matplotlib notebook

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title("3 component PCA", fontsize=30)

    # plot first k players' info along principal components
    k = 4000
    ax.scatter(
        principalDf.loc[:k, "principal component 1"],
        principalDf.loc[:k, "principal component 2"],
        principalDf.loc[:k, "principal component 3"],
        s=1,
    )

    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_zlabel("Principal Component 3", fontsize=15)
    plt.show()

    return principalDf


def kmeans_vis(Df):
    """
    Plot the 3-class k-means result on input dataset
    """
    data_pca = scale(Df)
    k_means = KMeans(n_clusters=3)
    k_means.fit(data_pca)
    k_means_predicted = k_means.predict(data_pca)
    plt.figure("K-Means on 3 component analysis", figsize=(7, 7))
    ax_2 = plt.axes(projection="3d")
    ax_2.scatter(
        data_pca[:1500, 0],
        data_pca[:1500, 1],
        data_pca[:1500, 2],
        c=k_means_predicted[:1500],
        cmap="Set2",
        s=20,
    )
    ax_2.set_title("K-means on PCA result", fontsize=20)
    ax_2.set_xlabel("Principal Component 1", fontsize=15)
    ax_2.set_ylabel("Principal Component 2", fontsize=15)
    ax_2.set_zlabel("Principal Component 3", fontsize=15)
    plt.show()
