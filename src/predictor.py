import pandas as pd
import math
from .elo import RugbyEloSystem  # Imports the ELO engine from the same folder

class TeamStats:
    """
    Tracks the Exponential Weighted Moving Averages (EWMA) for a team.
    Tracks: Attack (Points Scored), Defense (Points Conceded), and Venue Form.
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        # Initial Assumptions (Average Team stats)
        self.att_ewma = 25.0  # Assumes avg team scores 25 points
        self.def_ewma = 25.0  # Assumes avg team concedes 25 points
        self.home_margin_ewma = 5.0  # Assumes avg home win by 5
        self.away_margin_ewma = -5.0 # Assumes avg away loss by 5

    def update(self, pts_scored, pts_conceded, margin, is_home):
        """
        Updates the weighted averages with the latest match result.
        """
        # Update Attack/Defense (EWMA)
        self.att_ewma = (self.alpha * pts_scored) + ((1 - self.alpha) * self.att_ewma)
        self.def_ewma = (self.alpha * pts_conceded) + ((1 - self.alpha) * self.def_ewma)
        
        # Update Venue Specific Form (Home or Away)
        if is_home:
            self.home_margin_ewma = (self.alpha * margin) + ((1 - self.alpha) * self.home_margin_ewma)
        else:
            self.away_margin_ewma = (self.alpha * margin) + ((1 - self.alpha) * self.away_margin_ewma)


class RugbyPredictor:
    """
    The main engine that combines ELO (Context) and TeamStats (Performance)
    to predict scorelines.
    """
    def __init__(self, elo_scale=0.05):
        self.stats = {}  # Dictionary to store TeamStats objects
        self.elo_engine = RugbyEloSystem() # Internal ELO engine to track context
        self.elo_scale = elo_scale # Scale factor: 20 ELO diff ≈ 1 Point adjustment

    def _get_stats(self, team):
        # Helper to get or create stats for a team
        if team not in self.stats:
            self.stats[team] = TeamStats()
        return self.stats[team]

    def train_model(self, df):
        """
        Replays the entire history to build up the stats state.
        Critically, it calculates 'Adjusted Points' based on the opponent strength.
        """
        print("Training Prediction Model (EWMA + ELO Context)...")
        
        # Sort by date to ensure chronological training
        df = df.sort_values(by='Date').reset_index(drop=True)

        for _, row in df.iterrows():
            h_team, a_team = row['Home_Team'], row['Away_Team']
            s_h, s_a = row['Home_Score'], row['Away_Score']
            
            # 1. Get PRE-MATCH ratings (Context)
            # We need to know how strong the opponent was *before* this game
            elo_h = self.elo_engine.get_rating(h_team)
            elo_a = self.elo_engine.get_rating(a_team)
            
            # 2. Calculate "Adjusted" Performance
            # If I scored 20 vs a strong team, it's worth more than 20 vs a weak team.
            # Adjustment = (OpponentELO - 1500) * Scale
            h_adj_factor = (elo_a - 1500) * self.elo_scale
            a_adj_factor = (elo_h - 1500) * self.elo_scale
            
            # Adjusted Attack Score
            h_att_perf = s_h + h_adj_factor
            a_att_perf = s_a + a_adj_factor
            
            # Adjusted Defense Score (Conceding vs strong is forgivable)
            h_def_perf = s_a - a_adj_factor 
            a_def_perf = s_h - h_adj_factor
            
            # 3. Update EWMA Stats
            h_stats = self._get_stats(h_team)
            a_stats = self._get_stats(a_team)
            
            h_stats.update(h_att_perf, h_def_perf, (s_h - s_a), is_home=True)
            a_stats.update(a_att_perf, a_def_perf, (s_a - s_h), is_home=False)
            
            # 4. Update ELO Engine (Move history forward)
            self.elo_engine.update_match(
                h_team, a_team, s_h, s_a, 
                row['Competition'], row['Neutral_Venue']
            )

    def predict_match(self, home, away, is_neutral=False):
        """
        Predicts the scoreline for a future match.
        Returns a dictionary with the prediction details.
        """
        if home not in self.stats or away not in self.stats:
            return None
        
        h_stats = self.stats[home]
        a_stats = self.stats[away]
        
        # 1. Get Current ELOs for Context Adjustment
        elo_h = self.elo_engine.get_rating(home)
        elo_a = self.elo_engine.get_rating(away)
        
        # 2. Calculate Base Expected Scores
        # Formula: (My Avg Adjusted Attack + Opponent Avg Adjusted Defense) / 2
        # Then "De-Normalize" by subtracting the opponent strength adjustment
        
        # Home Expectation
        h_adj_factor = (elo_a - 1500) * self.elo_scale
        pred_h_pts = ((h_stats.att_ewma + a_stats.def_ewma) / 2) - h_adj_factor
        
        # Away Expectation
        a_adj_factor = (elo_h - 1500) * self.elo_scale
        pred_a_pts = ((a_stats.att_ewma + h_stats.def_ewma) / 2) - a_adj_factor
        
        # 3. Apply Dynamic Home Advantage
        # HA = (Home Team's Avg Home Margin - Away Team's Avg Away Margin) / 2
        if not is_neutral:
            # Note: Away Margin is usually negative. So we subtract the negative (adding points).
            dynamic_ha = (h_stats.home_margin_ewma - a_stats.away_margin_ewma) / 2
            pred_h_pts += dynamic_ha
        
        # 4. Final Formatting
        score_h = int(round(max(0, pred_h_pts))) # Prevent negative scores
        score_a = int(round(max(0, pred_a_pts)))
        winner = home if score_h > score_a else away
        margin = abs(score_h - score_a)
        
        # Calculate Win Confidence (using ELO probability)
        # We assume standard HA of 60 for the probability calc if not neutral
        prob_ha = 0 if is_neutral else 60
        prob_h = 1 / (1 + 10 ** ((elo_a - (elo_h + prob_ha)) / 400))
        conf = prob_h if prob_h > 0.5 else (1 - prob_h)

        return {
            'Home': home,
            'Away': away,
            'Pred_Home_Score': score_h,
            'Pred_Away_Score': score_a,
            'Winner': winner,
            'Margin': margin,
            'Confidence': f"{conf*100:.1f}%",
            'Total_Points': score_h + score_a
        }
