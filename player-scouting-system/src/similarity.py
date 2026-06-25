import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_prep import load_data, clean_data, prepare_features


def build_model(position="FW"):

    df = load_data()

    df = clean_data(df, position)

    df, X_scaled, scaler = prepare_features(df, position)

    sim_matrix = cosine_similarity(X_scaled)

    return df, sim_matrix



def get_similar_players(df, sim_matrix, player_name, top_n=5):

    player_row = df[df["Player"] == player_name]

    if player_row.empty:
        print("Player not found")
        return None

    idx = player_row.index[0]

    scores = list(enumerate(sim_matrix[idx]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # remove the player itself
    scores = scores[1:top_n + 1]

    similar_indices = [i for i, _ in scores]
    similarity_scores = [round(score, 3) for _, score in scores]


    # some basic stats
    
    result = df.iloc[similar_indices][
        [
            "Player",
            "Squad",
            "Age",
            "Pos",
            "Gls",
            "Ast",
            "Sh",
            "SoT"
        ]
    ].copy()

    result["Similarity Score"] = similarity_scores

    return result


# test

if __name__ == "__main__":

    for pos in ["FW", "MF", "DF", "GK"]:

        print("\n=========================")
        print("POSITION:", pos)

        df, sim_matrix = build_model(pos)

        print(df["Player"].head(10))

        player = df["Player"].iloc[0]

        print(f"\nFinding similar players to: {player}\n")

        print(
            get_similar_players(df, sim_matrix, player)
        )