import nflreadpy as nfl
import numpy as np
import pandas as pd
import os

def fetch_data(years: list, positions=['QB', 'RB', 'WR', 'TE']) -> pd.DataFrame:
    """
    Fetches weekly player data for a list of specified years.
    """
    print(f"Fetching weekly data for seasons: {years}...")
    
    # We will store the DataFrame for each year in this list.
    all_data_list = []
    
    for year in years:
        try:
            # import_weekly_data - week-by-week stats for every player.
            weekly_df = nfl.load_player_stats(seasons=[year]).to_pandas()

            # Rename 'team' to 'recent_team' early so we can check it
            if 'team' in weekly_df.columns:
                weekly_df = weekly_df.rename(columns={'team': 'recent_team'})

            # Find 'position' feature and drop all rows with positions we dont care about
            if 'position' in weekly_df.columns:
                weekly_df = weekly_df[weekly_df['position'].isin(positions)]
            elif 'position_group' in weekly_df.columns:
                 weekly_df = weekly_df[weekly_df['position_group'].isin(positions)]

            # Fetch the second DataFrame (snaps)
            snaps_df = nfl.load_snap_counts(seasons=[year]).to_pandas()

            # Filter Snap Counts Immediately
            if 'position' in snaps_df.columns:
                snaps_df = snaps_df[snaps_df['position'].isin(positions)]

            # Merge the two DataFrames
            # These are the key columns from the LEFT DataFrame (weekly_df)
            left_join_cols = ['player_display_name', 'season', 'week', 'opponent_team']
            
            # These are the *corresponding* key columns from the RIGHT DataFrame (snaps_df)
            right_join_cols = ['player', 'season', 'week', 'opponent']

            # Select specific snap columns to avoid duplicates
            snaps_cols_to_keep = right_join_cols + ['offense_snaps', 'offense_pct', 'defense_snaps', 'st_snaps']

            # Ensure snap columns exist before merging
            available_snap_cols = [c for c in snaps_cols_to_keep if c in snaps_df.columns]
            
            # We do a 'left' merge.
            # It keeps *all* the rows from our main 'weekly_df' and only adds snap counts where they exist.
            combined_df = pd.merge(
                weekly_df,
                snaps_df[available_snap_cols],
                left_on=left_join_cols,
                right_on=right_join_cols,
                how='left'
            )
            
            all_data_list.append(combined_df)
            print(f"  Successfully fetched data for {year}.")
        except Exception as e:
            print(f"  Error fetching data for {year}: {e}")
            
    if not all_data_list:
        print("No data was fetched. Exiting.")
        return pd.DataFrame()

    # Combine all the yearly data into one big DataFrame.
    full_data = pd.concat(all_data_list, ignore_index=True)
    print("All data has been combined.")
    return full_data

def parse_data(df: pd.DataFrame, positions: list) -> pd.DataFrame:
    """
    Parses and cleans the raw data to make it usable for the model.
    This is where you'll do most of your "feature engineering" later.
    """
    print("Parsing data...")

    # nflreadpy/nflfastR uses different names than the old library.
    # We map them back to what the model expects.
    rename_map = {
        'team': 'recent_team',             # The biggest one
        'passing_interceptions': 'interceptions',
        'sacks_suffered': 'sacks',         # For QBs
        'sack_yards_lost': 'sack_yards'
    }
    df = df.rename(columns=rename_map)
    
    # Filter for relevant positions
    parsed_df = df[df['position'].isin(positions)].copy()
    
    # Select only the columns we care about.
    columns_to_keep = ['player_id', 'player_name', 'player_display_name', 'position', 'recent_team', 'season', 'week', 'opponent_team', 'offense_snaps',
     'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards', 'sack_fumbles', 
     'sack_fumbles_lost', 'passing_air_yards', 'passing_yards_after_catch', 'passing_first_downs', 
     'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 'rushing_first_downs',
     'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards', 
     'receiving_yards_after_catch', 'receiving_first_downs', 'target_share', 'air_yards_share', 
     'fantasy_points', 'fantasy_points_ppr']

    # Check for which columns *actually* exist
    actual_cols_to_keep = [col for col in columns_to_keep if col in df.columns]
    missing_cols = set(columns_to_keep) - set(actual_cols_to_keep)
    
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {missing_cols}")

    # Filter the DataFrame to only these columns
    clean_df = df[actual_cols_to_keep].copy()
    
    # Replace NaNs in numeric columns with 0
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    clean_df[numeric_cols] = clean_df[numeric_cols].fillna(0)
    
    print(f"DataFrame has been parsed. Kept {len(actual_cols_to_keep)} columns.")
    print("Filled all missing numeric values with 0.")
    
    return clean_df

def save_data(df: pd.DataFrame, filepath: str):
    """
    Saves the final processed DataFrame to a local CSV file.
    """
    print(f"Saving data to {filepath}...")
    
    # Create the output directory (e.g., 'data/') if it doesn't exist.
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV.
    # index=False is important to avoid saving the pandas index as a column.
    df.to_csv(filepath, index=False)
    
    print("Data successfully saved!")

def run_data_pipeline(years_to_fetch=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], positions=['QB', 'RB', 'WR', 'TE'], output_dir='data'):
    """
    Main function to run the data pipeline.
    """
    OUTPUT_FILENAME = os.path.join(output_dir, 'nfl_weekly_data.csv')
    
    raw_data = fetch_data(years=years_to_fetch, positions=positions)

    if not raw_data.empty:
        parsed_data = parse_data(df=raw_data, positions=positions)
        save_data(df=parsed_data, filepath=OUTPUT_FILENAME)