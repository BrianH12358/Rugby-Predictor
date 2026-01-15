import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CLASS DEFINITION
# ==========================================
class RugbyEloSystem:
    def __init__(self, k_factor=20, home_advantage=60, start_rating=1500):
        self.k = k_factor
        self.ha = home_advantage
        self.start_rating = start_rating
        self.ratings = {} 
        
        # DEFINED WEIGHTS
        self.comp_weights = {
            'Champions Cup': 1.5,
            'Challenge Cup': 1.25,
            # Domestic leagues default to 1.0 automatically
        }

    def get_rating(self, team):
        return self.ratings.get(team, self.start_rating)

    def calculate_margin_multiplier(self, score_diff):
        return math.log(abs(score_diff) + 1)

    def expected_result(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_match(self, home_team, away_team, score_home, score_away, competition, is_neutral=False):
        r_home = self.get_rating(home_team)
        r_away = self.get_rating(away_team)

        # Determine Actual Result
        if score_home > score_away:
            actual_result = 1
        elif score_home == score_away:
            actual_result = 0.5
        else:
            actual_result = 0

        # Determine Expected Result
        effective_ha = 0 if is_neutral else self.ha
        we_home = self.expected_result(r_home + effective_ha, r_away)

        # Calculate Multipliers
        mov_mult = self.calculate_margin_multiplier(score_home - score_away)
        comp_mult = self.comp_weights.get(competition, 1.0)
        k_final = self.k * comp_mult

        # Calculate Point Change
        point_change = k_final * mov_mult * (actual_result - we_home)

        # Apply Away Win Bonus
        if actual_result == 0: 
            point_change *= 1.1

        # Update Ratings Dictionary
        self.ratings[home_team] = r_home + point_change
        self.ratings[away_team] = r_away - point_change

        return {
            'Change': round(point_change, 2),
            'New_Rating_H': round(self.ratings[home_team], 1)
        }
