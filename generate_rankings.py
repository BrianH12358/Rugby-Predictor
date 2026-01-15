import pandas as pd
import os
from src.elo import RugbyEloSystem

# Use relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'matches.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'outputs', 'final_rankings.csv')

def main():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, encoding='latin1')
    
    # Cleaning
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['Home_Score', 'Away_Score'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Initialize
    elo = RugbyEloSystem()
    
    # Process
    print("Processing matches...")
    for _, row in df.iterrows():
        elo.update_match(
            row['Home_Team'], row['Away_Team'], 
            row['Home_Score'], row['Away_Score'], 
            row['Competition'], row['Neutral_Venue']
        )
        
    # Save
    ranking_df = pd.DataFrame(list(elo.ratings.items()), columns=['Team', 'Elo_Rating'])
    ranking_df = ranking_df.sort_values(by='Elo_Rating', ascending=False)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True) # Ensure folder exists
    ranking_df.to_csv(OUTPUT_PATH, index=False, encoding='latin1')
    print(f"Rankings saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
