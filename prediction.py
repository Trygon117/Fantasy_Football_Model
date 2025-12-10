import pandas as pd
import nflreadpy as nfl
from fantasy_model import FantasyModel

def predict_upcoming_week(model: FantasyModel, season: int, week: int):
    """
    Generates predictions for a specific future week using nflreadpy.
    """
    print(f"--- Predicting {season} Week {week} ---")
    
    # Force reload from disk
    print("Forcing a clean data reload to purge ghost rows...")
    model.load_data() 
    
    # This deletes any rows that match the target season/week BEFORE we append the new ones.
    print(f"Purging any existing data for {season} Week {week}...")
    mask = (model.raw_weekly_data['season'] == season) & (model.raw_weekly_data['week'] == week)
    model.raw_weekly_data = model.raw_weekly_data[~mask]
    
    # Fetch Schedule & Rosters
    try:
        schedule = nfl.load_schedules(seasons=[season]).to_pandas()
        current_week_games = schedule[schedule['week'] == week]
        
        # Load rosters and fix ID names
        rosters = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
        if 'gsis_id' in rosters.columns:
            rosters = rosters.rename(columns={'gsis_id': 'player_id'})
            
    except Exception as e:
        print(f"Error fetching NFL data: {e}")
        return

    # Deduplicate Rosters
    if 'week' in rosters.columns:
        rosters = rosters.sort_values(['season', 'week'])
        rosters = rosters.drop_duplicates(subset=['player_id'], keep='last')
    else:
        rosters = rosters.drop_duplicates(subset=['player_id'])

    # Build Matchup Map
    team_to_opponent = {}
    for _, game in current_week_games.iterrows():
        team_to_opponent[game['home_team']] = game['away_team']
        team_to_opponent[game['away_team']] = game['home_team']
        
    # Create Future DataFrame
    RELEVANT_POSITIONS = ['QB', 'RB', 'WR', 'TE']
    rosters = rosters[rosters['position'].isin(RELEVANT_POSITIONS)]
    
    # Only keep players playing this week
    upcoming_players = rosters[rosters['team'].isin(team_to_opponent.keys())].copy()
    
    future_data = pd.DataFrame()
    future_data['player_id'] = upcoming_players['player_id']
    
    # Handle Name Differences
    if 'full_name' in upcoming_players.columns:
        future_data['player_name'] = upcoming_players['full_name']
        future_data['player_display_name'] = upcoming_players['full_name']
    else:
        future_data['player_name'] = upcoming_players['player_name']
        future_data['player_display_name'] = upcoming_players['player_name']
        
    future_data['position'] = upcoming_players['position']
    future_data['recent_team'] = upcoming_players['team']
    future_data['season'] = season
    future_data['week'] = week
    future_data['opponent_team'] = future_data['recent_team'].map(team_to_opponent)
    
    # Fill Stats with 0
    stat_cols = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 
                 'sacks', 'carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets', 
                 'receiving_yards', 'receiving_tds', 'fantasy_points_ppr', 'offense_snaps']
    
    for col in stat_cols:
        future_data[col] = 0
        
    # Append & Engineer
    # Now safe to append because we purged the old version of this week in Step 1
    model.raw_weekly_data = pd.concat([model.raw_weekly_data, future_data], ignore_index=True)
        
    # Process
    model.engineer_features()
    model.preprocess()
    
    # Predict
    full_features = model.feature_engineered_data
    X_pred_rows = full_features[(full_features['season'] == season) & (full_features['week'] == week)].copy()
    
    non_features = ['fantasy_points_ppr', 'fantasy_points', 'player_id', 'player_name', 
                    'player_display_name', 'season', 'week']
    X_pred = X_pred_rows.drop(columns=non_features, errors='ignore')
    
    # Match Columns
    try:
        required_features = model.trained_model.get_booster().feature_names
        if required_features:
            missing_cols = set(required_features) - set(X_pred.columns)
            for c in missing_cols:
                X_pred[c] = 0
            X_pred = X_pred[required_features]
    except Exception as e:
        print(f"Warning: Could not verify column order ({e})")
    
    preds = model.trained_model.predict(X_pred)
    X_pred_rows['projected_points'] = preds
    
    # Restore Readable Columns
    cols_to_restore = ['player_id', 'player_display_name', 'position', 'recent_team', 'opponent_team']
    
    raw_subset = model.raw_weekly_data[
        (model.raw_weekly_data['season'] == season) & 
        (model.raw_weekly_data['week'] == week)
    ][cols_to_restore]
    
    raw_subset = raw_subset.drop_duplicates(subset=['player_id'])
    
    final_view = pd.merge(
        X_pred_rows[['player_id', 'projected_points']], 
        raw_subset,
        on='player_id',
        how='left'
    )
    
    final_view = final_view.sort_values('projected_points', ascending=False)
    final_view = final_view[['player_display_name', 'position', 'recent_team', 'opponent_team', 'projected_points']]
    
    return final_view