
# Player Scouting System

A football analytics and scouting platform built with Python and Streamlit.

The application allows users to search players, discover statistically similar players, compare player profiles, and visualize performance through percentile-based radar charts. The player stats come from a database found on kaggle (https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2025-2026).

## Features

* Player search with autocomplete
* Player profile dashboard
* Position-specific scouting models
* Similar player recommendations
* Player comparison tool
* Percentile radar visualizations

## Supported Positions

* Forwards (FW)
* Midfielders (MF)
* Defenders (DF)
* Goalkeepers (GK)

Each position uses its own statistical profile and weighted similarity model.

## How It Works

1. Player data is cleaned and filtered by position.
2. Relevant statistics are selected for each position.
3. Features are standardized using `StandardScaler`.
4. Position-specific weights are applied to scouting metrics.
5. Cosine Similarity is used to identify statistically similar players.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* SciPy


## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run src/app.py
```

## Possible Future Improvements

* Expected Goals (xG) integration
* Transfer recommendation engine
* League and age-based filtering
* Team scouting reports
* Multi-season analysis

