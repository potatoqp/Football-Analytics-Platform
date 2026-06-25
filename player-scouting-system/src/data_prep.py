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
        "Fld",      # fouls drawn
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

POSITION_WEIGHTS = {

    "FW": {
        "Gls": 3.0,
        "Ast": 2.0,
        "Sh": 2.5,
        "SoT": 2.5,
        "G/Sh": 2.0,
        "G/SoT": 2.0,
        "Crs": 1.0,
        "TklW": 0.5,
        "Int": 0.5,
        "CrdY": 0.2
    },

    "MF": {
        "Ast": 3.0,
        "Compl": 3.0,
        "Fld": 2.0,
        "Crs": 2.0,
        "Int": 1.5,
        "TklW": 1.5,
        "Sh": 1.5,
        "SoT": 1.5,
        "Fls": 0.5,
        "CrdY": 0.5
    },

    "DF": {
        "Int": 3.0,
        "TklW": 3.0,
        "Fls": 1.5,
        "Fld": 1.0,
        "CrdY": 1.0,
        "Off": 0.2,
        "Crs": 1.0,
        "Sh": 0.5
    },

    "GK": {
        "Saves": 3.0,
        "Save%": 4.0,
        "CS": 2.5,
        "GA": 2.0,
        "CS%": 2.0,
        "PKsv": 1.5,
        "PKA": 0.5
    }
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

    weights = [
        POSITION_WEIGHTS[position][feature]
        for feature in features
    ]

    X_scaled = X_scaled * weights

    print(
        f"Features ready ({position}):",
        X.shape
    )

    print(
        f"Weights applied ({position}):",
        weights
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