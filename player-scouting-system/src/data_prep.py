import pandas as pd
from sklearn.preprocessing import StandardScaler


POSITION_FEATURES = {

    
    "FW": [
        "Gls",      # goals scored
        "Ast",      # assists
        "Sh",       # shots
        "SoT",      # shots on target
        "Crs",      # crosses
        "G/Sh",     # goals per shot
        "G/SoT",    # goals per shot on target
        "Crs",      # crosses
        "TklW",     # tackles won
        "Int",      # interceptions
        "CrdY"      # yellow cards
    ],

    
    "MF": [
        "Ast",      # assists
        "Crs",      # crosses
        "Int",      # interceptions
        "TklW",     # tackles won
        "Compl",    # completed passes
        "Fld",      # failed passes
        "Fls",      # fouls committed
        "Sh",       # shots
        "SoT",      # shots on target
        "CrdY"      # yellow cards
    ],

    
    "DF": [
        "Int",      # interceptions
        "TklW",     # tackles won
        "Fls",      # fouls committed
        "Fld",      # fouls drawn
        "CrdY",     # yellow cards
        "Off",      # offsides
        "Crs",      # crosses
        "Sh"        # shots
    ],

    
    "GK": [
        "GA",        # goals conceded
        "Saves",     # saves made
        "Save%",     # save percentage
        "CS",        # clean sheets
        "CS%",       # clean sheet %
        "PKsv",      # penalties saved
        "PKA"        # penalties faced
    ]
}



def load_data(path="data/players.csv"):
    df = pd.read_csv(path)

    print("Data loaded:", df.shape)

    return df



def clean_data(df, position):

    df = df.copy()

    df = df[df["Pos"].notna()]

    # IMPORTANT: GK is exact match, others are contains
    if position == "GK":
        df = df[df["Pos"].str.contains("GK", na=False)]
    else:
        df = df[df["Pos"].str.contains(position, na=False)]

    df = df[df["MP"] >= 10]

    df = df.dropna(
        subset=[
            "Player",
            "Pos",
            "MP"
        ]
    )

    print(
        f"After cleaning ({position}):",
        df.shape
    )

    return df



def prepare_features(df, position):

    df = df.copy()

    features = POSITION_FEATURES[position]

    df = df.dropna(subset=features)

    df = df.reset_index(drop=True)

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(
        f"Features ready ({position}):",
        X.shape
    )

    return df, X_scaled, scaler


# test

if __name__ == "__main__":

    for pos in ["FW", "MF", "DF", "GK"]:

        print("\n=========================")
        print("POSITION:", pos)

        df = load_data()

        df = clean_data(df, pos)

        df, X_scaled, scaler = prepare_features(df, pos)

        print(df[["Player", "Squad", "Pos"]].head())