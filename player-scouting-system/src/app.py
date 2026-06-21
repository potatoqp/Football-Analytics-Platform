import streamlit as st

from similarity import build_model, get_similar_players

@st.cache_data
def load_model():
    df, sim_matrix = build_model()
    return df, sim_matrix


df, sim_matrix = load_model()

st.title("⚽ Player Scouting System")
st.write("Find players with similar statistical profiles")


player_list = sorted(df["Player"].unique())

selected_player = st.selectbox(
    "Search or select a player:",
    player_list
)

if st.button("🔍 Find Similar Players"):

    results = get_similar_players(
        df,
        sim_matrix,
        selected_player,
        top_n=5
    )

    if results is not None:
        st.subheader(f"Players similar to {selected_player}")

        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True
        )