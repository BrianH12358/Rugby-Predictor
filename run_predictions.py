mport pandas as pd
import os
from src.predictor import RugbyPredictor

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'matches.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'outputs', 'predictions.csv')

def main():
    # 1. Load Data
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, encoding='latin1')
    
    # Cleaning
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['Home_Score', 'Away_Score'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    # 2. Train the Model (Replay History)
    predictor = RugbyPredictor()
    predictor.train_model(df)
    
    # 3. Define Future Fixtures to Predict
    # (In a real scenario, you might load these from a 'fixtures.csv' file)
    upcoming_fixtures = [
          ('Section Paloise', 'Bulls', False),
          ('Bath Rugby', 'Edinburgh', False),
          ('The Sharks', 'ASM Clermont Auvergne', False),
          ('Stormers', 'Leicester Tigers', False),
          ('Aviron Bayonnais', 'Leinster', False),
          ('Stade Toulousain', 'Sale Sharks', False),
          ('Munster', 'Castres Olympique', False),
          ('Gloucester', 'RC Toulonnais', False),
          ('Bristol Bears', 'Union Bordeaux Bègles', False),
          ('Northampton Saints', 'Scarlets', False),
          ('Stade Rochelais', 'Harlequins', False),
          ('Glasgow', 'Saracens', False),
          ('Munster', 'Northampton Saints', False)
    ]

    print("\n--- Generating Predictions ---")
    results = []
    for home, away, neutral in upcoming_fixtures:
        pred = predictor.predict_match(home, away, is_neutral=neutral)
        if pred:
            results.append(pred)
            print(f"{home} vs {away}: {pred['Pred_Home_Score']} - {pred['Pred_Away_Score']} ({pred['Winner']})")
        else:
            print(f"Skipping {home} vs {away}: Team not found.")

    # 4. Save to CSV
    if results:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        print(f"\nPredictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
