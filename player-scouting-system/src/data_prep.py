import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path="data/players.csv"):
    df = pd.read_csv(path)
    print("Data loaded:", df.shape)
    return df


def clean_data(df):
    df = df.copy()

    # keep only forwards 
    df = df[df["Pos"].notna()]
    df = df[df["Pos"].str.contains("FW", na=False)]

    # remove low sample players
    df = df[df["MP"] >= 10]

    # drop only essential missing columns
    df = df.dropna(subset=["Player", "Pos", "MP"])

    print("After cleaning:", df.shape)
    return df



FEATURES = [
    # finishing / attacking output
    "Gls",      #goals
    "Ast",      #assists
    "Sh",       #shots
    "SoT",      #shots on target
    "G/Sh",     #goals per shot
    "G/SoT",    #goals per shot on target   

    # passing / creativity
    "Crs",      #crosses

    # defensive contribution 
    "TklW",     #tackles won
    "Int",      #interceptions

    # discipline / mistakes
    "CrdY",     #yellow cards
    "CrdR",     #red cards
]


def prepare_features(df):
    df = df.copy()

    # remove rows with missing feature values only
    df = df.dropna(subset=FEATURES)

    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Features ready:", X.shape)
    return df, X_scaled, scaler


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df, X_scaled, scaler = prepare_features(df)

    print("\nSample players:")
    print(df[["Player", "Squad", "Pos", "Gls", "Ast"]].head())