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

player_data = df[df["Player"] == selected_player].iloc[0]

st.subheader("Player Profile")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Club:** {player_data['Squad']}")
    st.write(f"**Age:** {player_data['Age']}")

with col2:
    st.write(f"**Position:** {player_data['Pos']}")
    st.write(f"**League:** {player_data['Comp']}")

st.write("---")

st.write(f"⚽ Goals: {player_data['Gls']}")
st.write(f"🎯 Assists: {player_data['Ast']}")
st.write(f"🥅 Shots: {player_data['Sh']}")
st.write(f"✅ Shots on Target: {player_data['SoT']}")

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