# European Rugby ELO & Prediction Engine

A data science project that ranks European Rugby clubs (URC, Premiership, Top 14) and predicts match outcomes using a custom ELO engine and Exponential Weighted Moving Averages (EWMA).

## Features
- **Continuous ELO System:** Accounts for Margin of Victory (MoV).
- **Cross-Border Weighting:** Champions Cup games carry 1.5x weighting to bridge league capabilities.
- **Score Predictor:** Combines ELO strength with recent attack/defense form to predict scorelines.

## Usage
1. Install dependencies:
   `pip install -r requirements.txt`

2. Place your match data in `data/raw/matches.csv`.

3. Generate Rankings:
   `python generate_rankings.py`

4. Run Predictions:
   `python run_predictions.py`
