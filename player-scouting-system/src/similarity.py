import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_prep import load_data, clean_data, prepare_features


def build_model():
    df = load_data()
    df = clean_data(df)
    df, X_scaled, scaler = prepare_features(df)

    #similarity matrix (player vs player)
    sim_matrix = cosine_similarity(X_scaled)

    return df, sim_matrix


def get_similar_players(df, sim_matrix, player_name, top_n=5):
    if player_name not in df["Player"].values:
        print("Player not found")
        return None

    idx = df[df["Player"] == player_name].index[0]

    scores = list(enumerate(sim_matrix[idx]))

    #sort by similarity score (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    #skip first one (player itself)
    scores = scores[1:top_n+1]

    similar_indices = [i for i, _ in scores]

    result = df.iloc[similar_indices][
        ["Player", "Squad", "Age", "Pos", "Gls", "Ast", "Sh", "SoT"]
    ]

    return result


if __name__ == "__main__":
    df, sim_matrix = build_model()

    print("\nAvailable players (sample):")
    print(df["Player"].head(10))

    player = df["Player"].iloc[0]  # pick first player for test

    print(f"\nFinding players similar to: {player}\n")

    similar = get_similar_players(df, sim_matrix, player)

    print(similar)