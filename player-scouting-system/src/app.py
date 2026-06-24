import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore



from similarity import build_model, get_similar_players

@st.cache_data
def load_model():
    df, sim_matrix = build_model()
    return df, sim_matrix

def create_percentile_radar(player_row, df):

    stat_mapping = {
        "Goals": "Gls",
        "Assists": "Ast",
        "Shots": "Sh",
        "Shots On Target": "SoT",
        "Crosses": "Crs",
        "Interceptions": "Int",
        "Tackles Won": "TklW"
    }

    categories = list(stat_mapping.keys())

    values = []

    for stat_name, column in stat_mapping.items():

        percentile = percentileofscore(
            df[column],
            player_row[column],
            kind="rank"
        )

        values.append(percentile)

    values += values[:1]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(categories),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    ax.set_ylim(0, 100)

    ax.set_title(
        f"{player_row['Player']} Percentile Profile",
        pad=20
    )

    return fig


df, sim_matrix = load_model()

st.title("⚽ Player Scouting System")
st.write("Find players with similar statistical profiles")


player_list = sorted(df["Player"].unique())

selected_player = st.selectbox(
    "Search or select a player:",
    player_list,
    index=None,
    placeholder="Type to search a player..."
)

if selected_player is None:
    st.stop()

player_data = df[df["Player"] == selected_player].iloc[0]

tab1, tab2, tab3 = st.tabs([
    "Player Profile",
    "Find Similar Players",
    "Compare Players"
])

with tab1:

    st.subheader("Player Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Club:** {player_data['Squad']}")
        st.write(f"**Age:** {player_data['Age']}")

    with col2:
        st.write(f"**Position:** {player_data['Pos']}")
        st.write(f"**League:** {player_data['Comp']}")

    st.write("---")

    st.write(f"Goals: {player_data['Gls']}")
    st.write(f"Assists: {player_data['Ast']}")
    st.write(f"Shots: {player_data['Sh']}")
    st.write(f"Shots on Target: {player_data['SoT']}")

    st.subheader("Player Radar")

    fig = create_percentile_radar(player_data, df)

    st.pyplot(fig)

with tab2:

    st.subheader(
        f"Players similar to {selected_player}"
    )

    results = get_similar_players(
        df,
        sim_matrix,
        selected_player,
        top_n=5
    )

    if results is not None:

        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True
        )

with tab3:

    st.subheader("Player Comparison")

    col1, col2 = st.columns(2)

    with col1:
        player_a = st.selectbox(
            "Player A",
            player_list,
            key="player_a",
            index = None,
            placeholder="Select Player A"
        )

    with col2:
        player_b = st.selectbox(
            "Player B",
            player_list,
            key="player_b",
            index = None,   
            placeholder="Select Player B"
        )

    if player_a is None or player_b is None:
        st.stop()

    if player_a == player_b:
        st.warning("Please select two different players")
        st.stop()

    if player_a and player_b:

        a = df[df["Player"] == player_a].iloc[0]
        b = df[df["Player"] == player_b].iloc[0]

        st.write("---")

    compare_stats = pd.DataFrame({
    "Stat": [
        "Goals",
        "Assists",
        "Shots",
        "Shots on Target",
        "Crosses",
        "Interceptions",
        "Tackles Won"
    ],
    player_a: [
        a["Gls"],
        a["Ast"],
        a["Sh"],
        a["SoT"],
        a["Crs"],
        a["Int"],
        a["TklW"]
    ],
    player_b: [
        b["Gls"],
        b["Ast"],
        b["Sh"],
        b["SoT"],
        b["Crs"],
        b["Int"],
        b["TklW"]
    ]
})
    
st.subheader("Visual Comparison")

stats = ["Gls", "Ast", "Sh", "SoT", "Crs", "Int", "TklW"]

max_vals = df[stats].max()

for stat in stats:

    st.write(f"**{stat}**")

    col1, col2 = st.columns(2)

    with col1:
        st.progress(float(a[stat]) / max_vals[stat])
        st.caption(player_a)

    with col2:
        st.progress(float(b[stat]) / max_vals[stat])
        st.caption(player_b) 

st.dataframe(compare_stats, use_container_width=True)

    
    
