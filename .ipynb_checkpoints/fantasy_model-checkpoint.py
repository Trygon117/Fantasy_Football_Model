import os
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt

class FantasyModel:
    """
    A class to encapsulate the entire fantasy football prediction pipeline,
    from data loading and feature engineering to model training and interpretation.
    """

    def __init__(self, data_filepath: str):
        """
        Initializes the model pipeline.
        
        Args:
            data_filepath (str): The path to the 'nfl_weekly_data.csv' file.
        """
        self.data_filepath = data_filepath
        
        # Raw data loaded from the CSV
        self.raw_weekly_data: pd.DataFrame = None
        
        # The data after feature engineering (e.g., seasonal stats)
        # This will be the direct input to our model.
        self.feature_engineered_data: pd.DataFrame = None
        
        # The features (X) and the value we want to predict (y)
        self.features: pd.DataFrame = None
        self.target: pd.Series = None
        
        # Data split into training and testing sets
        self.training_features: pd.DataFrame = None
        self.testing_features: pd.DataFrame = None
        self.training_target: pd.Series = None
        self.testing_target: pd.Series = None
        
        # The trained model and its explainer
        self.trained_model = None
        self.model_explainer = None
        self.feature_importance_values = None

    def load_data(self):
        """
        Loads the raw weekly data from the CSV file specified in self.data_filepath.
        Stores the data in self.raw_weekly_data.
        """
        try:
            self.raw_weekly_data = pd.read_csv(self.data_filepath)
            
            # It's also good practice to give yourself feedback in a pipeline.
            print(f"Successfully loaded data from {self.data_filepath}")
            print(f"Shape of raw data: {self.raw_weekly_data.shape}")

        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_filepath}")
            print("Please make sure 'nfl_weekly_data.csv' is in the 'data/' directory.")
            exit()
            
        except Exception as e:
            print(f"An unexpected error occurred while loading the data: {e}")
            exit()

    def perform_eda(self):
        """
        Performs Exploratory Data Analysis on self.raw_weekly_data.
        """
        print("--- Starting Exploratory Data Analysis (EDA) ---")

        # Make sure we have the data loaded
        if self.raw_weekly_data is None:
            print("Error: Data not loaded. Please run load_data() first.")
            return

        # --- Create a directory to save our plots ---
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory: {plot_dir}")

        # Define the positions we want to analyze
        # key fantasy positions and ignore noise (like 'K', 'P', 'FB').
        positions_to_plot = ['QB', 'RB', 'WR', 'TE']
        
        print(f"Data shape before EDA: {self.raw_weekly_data.shape}")
        
        # CHECK FOR MISSING DATA
        print("\n[EDA Part 1: Missing Data Analysis]")
        
        # Calculate missing values for every column
        missing_counts = self.raw_weekly_data.isnull().sum()
        
        # We only want to look at columns that actually have missing data.
        missing_counts = missing_counts[missing_counts > 0]
        
        if missing_counts.empty:
            print("No missing data found in any columns.")
        else:
            print("Missing data found. Plotting counts...")
            
            # --- Generate the missing data plot ---
            plt.figure(figsize=(12, 7)) # Make the plot wide enough for labels
            missing_counts.sort_values(ascending=False).plot(kind='bar')
            plt.title('Missing Data Counts by Feature', fontsize=16)
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Number of Missing Rows', fontsize=12)
            plt.tight_layout() # Helps fit the labels

            # Save the plot
            save_path = os.path.join(plot_dir, 'missing_data_counts.png')
            plt.savefig(save_path)
            print(f"Saved missing data plot to: {save_path}")

            # Show the graph in Jupyter
            #plt.show()

            # Close the plot to free up memory
            plt.close()

        # PLOT NUMERICAL FEATURES BY POSITION
        print("\n[EDA Part 2: Numerical Feature Box Plots]")
        
        for pos in positions_to_plot:
            print(f"\n--- Starting EDA for Position: {pos} ---")

            # 1. Filter the data for *only* this position
            pos_data = self.raw_weekly_data[self.raw_weekly_data['position'] == pos]
    
        
            # Make sure we are only plotting numerical features
            numeric_cols = pos_data.select_dtypes(include=[np.number])
        
            # 'season' and 'week' are *technically* numbers, but plotting them as a box plot isn't very useful.
            try:
                numeric_cols = numeric_cols.drop(columns=['season', 'week'])
            except KeyError:
                # In case they were already dropped or not present
                pass
            
            if numeric_cols.empty:
                print(f"No numerical columns found to plot for {pos}.")
                return

            print(f"Generating {pos} box plots for {len(numeric_cols.columns)} numerical features...")
        
            # Loop through each numerical column and create a box plot
            for column in numeric_cols.columns:
                plt.figure(figsize=(8, 6))
                # Create the box plot for the single column
                self.raw_weekly_data.boxplot(column=[column])
                
                plt.title(f'{pos} Box Plot for "{column}"', fontsize=16)
                plt.ylabel('Value', fontsize=12)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    
                # Save the plot with a unique name
                save_path = os.path.join(plot_dir, f'{pos}_boxplot_{column}.png')
                plt.savefig(save_path)
    
                # Show the graph in Jupyter
                #plt.show()
                
                # Close the plot to free up memory
                plt.close()

        print("--- EDA Complete ---")

    def engineer_features(self):
        """
        This is the most important function. It transforms the raw weekly
        data (self.raw_weekly_data) into a dataset ready for modeling.
        """
        print("--- Engineering Features ---")
        if self.raw_weekly_data is None:
            print("Error: Data not loaded. Please run load_data() first.")
            return

        # --- FILTER FOR RELEVANT POSITIONS ONLY ---
        RELEVANT_POSITIONS = ['QB', 'RB', 'WR', 'TE']

        # Filter the dataframe to keep only these positions
        self.raw_weekly_data = self.raw_weekly_data[
            self.raw_weekly_data['position'].isin(RELEVANT_POSITIONS)
        ].copy()

        print(f"Filtered data to {RELEVANT_POSITIONS} only. Shape: {self.raw_weekly_data.shape}")

        # Make sure the features are sorted correctly 
        self.raw_weekly_data = self.raw_weekly_data.sort_values(
            by=['player_id', 'season', 'week'], 
            ascending=[True, True, True]
        )

        # These are columns that we can use "as is"
        safe_columns = [
            'player_id', 'player_display_name', 'position', 'recent_team', 
            'season', 'week', 'opponent_team', 'offense_snaps', 'fantasy_points_ppr',
            'offense_snaps' # We will use 'offense_snaps' for our filter
        ]

        clean_data = self.raw_weekly_data[safe_columns].copy()

        # Define the list of stats we'll be engineering
        stat_columns = [
            'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions', 'sacks', 
            'passing_air_yards', 'passing_yards_after_catch',
            'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_fumbles',
            'receiving_air_yards', 'receiving_yards_after_catch',
            'target_share', 'air_yards_share', 'offense_snaps',
            'fantasy_points_ppr'
        ]

        grouping_keys = ['player_id', 'season']

        
        ### Current Season Stats ###

        
        print("Calculating 'Recent Form' (EWMA) and 'Season-to-Date' (Expanding) features...")

        for stat in stat_columns:
            # We read from the *raw* data
            current_stat_series = self.raw_weekly_data[stat]

            # Calculate a "Exponentially Weighted Moving Average"
            ewm_col_name = f'ewm_{stat}'

            # Calculate the EWMA for every row and add it to clean_data
            # We use the player id and season features to map from the current_stat_series
            # Apply the EWMA calculation to each group's stats
            # Shift the data down one row (to avoid leaking data)
            # .values ensures we get a clean array
            clean_data[ewm_col_name] = current_stat_series.groupby(
                [self.raw_weekly_data['player_id'], self.raw_weekly_data['season']], group_keys=True).apply(
                lambda x: x.ewm(span=5, adjust=False).mean().shift(1)).values

            
            # Calculate average performance "Season-to-Date"
            expanding_col_name = f'expanding_{stat}'

            # We use the player id and season features to map from the current_stat_series
            # Calculate the average performance for the current season using .expanding
            # Shift the data down one row (to avoid leaking data)
            # .values ensures we get a clean array
            clean_data[expanding_col_name] = current_stat_series.groupby(
                [self.raw_weekly_data['player_id'], self.raw_weekly_data['season']], group_keys=True).apply(
                lambda x: x.expanding().mean().shift(1)).values
            
        print("Finished 'Recent Form' and 'Season-to-Date' features.")

        
        ### DEPTH CHART RANKING ###
        
        print("Calculating Dual Depth Chart Ranks (Stable vs. Recent)...")

        ### Feature - Expanding ranked team position performance ###
        # Ranking based on expanding (Season-to-Date) average. This is robust to injuries.
        clean_data['depth_chart_rank_stable'] = clean_data.groupby(
            ['season', 'week', 'recent_team', 'position'])['expanding_fantasy_points_ppr'].rank(
            ascending=False, method='min')

        ### Feature - EWM ranked team position performance ###
        # Ranking based on EWM (Recent Form). This captures current hot streaks/demotions.
        clean_data['depth_chart_rank_recent'] = clean_data.groupby(
            ['season', 'week', 'recent_team', 'position'])['ewm_fantasy_points_ppr'].rank(
            ascending=False, method='min')

        ### Feature - Depth chart delta ###
        # A large positive delta means the player's current performance (recent) is
        # significantly *worse* than their stable status. (E.g., Star Player in a slump).
        # A large negative delta means the player is outperforming their stable status. (E.g., Rookie Ascending).
        clean_data['depth_chart_rank_delta'] = clean_data['depth_chart_rank_recent'] - clean_data['depth_chart_rank_stable']

        ### Feature - Is this player the #1 stable option? (Binary) ###
        clean_data['is_depth_chart_1_stable'] = (clean_data['depth_chart_rank_stable'] == 1).astype(int)

        ### Feature - Gap to Leader (using expanding ppr) ###
        max_points_on_team = clean_data.groupby(['season', 'week', 'recent_team', 'position'])['expanding_fantasy_points_ppr'].transform('max')
        clean_data['points_diff_from_team_leader_stable'] = max_points_on_team - clean_data['expanding_fantasy_points_ppr']

        
        ### DEFENSIVE MATCHUP FEATURES (Opponent Strength) ###
        
        print("Calculating 'Defense vs Position' stats...")

        # Calculate TOTAL points allowed by the defense per game/position
        # Group by Opponent, Season, Week, and Position to squash all WRs into one number
        defense_game_stats = self.raw_weekly_data.groupby(['opponent_team', 'season', 'week', 'position'])['fantasy_points_ppr'].sum().reset_index()
        
        # Rename for clarity
        defense_game_stats = defense_game_stats.rename(columns={'fantasy_points_ppr': 'game_total_allowed'})

        # Sort chronologically
        defense_game_stats = defense_game_stats.sort_values(['opponent_team', 'season', 'week'])

        # Calculate Expanding Average (Average POINTS PER GAME allowed)
        # We group by Opponent/Season/Position and average the WEEKLY TOTALS
        defense_game_stats['opponent_avg_points_allowed'] = defense_game_stats.groupby(
            ['opponent_team', 'season', 'position']
        )['game_total_allowed'].transform(lambda x: x.expanding().mean().shift(1))

        # Handle Week 1
        ## For Week 1  fill this with the global average for that position.
        global_pos_avg = defense_game_stats.groupby('position')['game_total_allowed'].transform('mean')
        defense_game_stats['opponent_avg_points_allowed'] = defense_game_stats['opponent_avg_points_allowed'].fillna(global_pos_avg)

        # Merge back into the main data
        # Note: We merge on ['opponent_team', 'season', 'week', 'position']
        clean_data = pd.merge(
            clean_data,
            defense_game_stats[['opponent_team', 'season', 'week', 'position', 'opponent_avg_points_allowed']],
            on=['opponent_team', 'season', 'week', 'position'],
            how='left'
        )
        
        print("Finished Defensive Matchup features.")


        ### OFFENSIVE ENVIRONMENT FEATURES (Team Strength) ###
        
        print("Calculating 'Team Offense' stats...")

        # Create a view to calculate Total Team Points per Week
        # We group by Team, Season, and Week to get the "Game Total" for that offense
        team_weekly_stats = self.raw_weekly_data.groupby(['recent_team', 'season', 'week'])['fantasy_points_ppr'].sum().reset_index()
        
        # Rename the column so we don't confuse it with individual points
        team_weekly_stats = team_weekly_stats.rename(columns={'fantasy_points_ppr': 'team_total_points'})

        # Sort by Date to ensure history is correct
        team_weekly_stats = team_weekly_stats.sort_values(['recent_team', 'season', 'week'])

        # Calculate Expanding Average
        team_weekly_stats['team_avg_points_scored'] = team_weekly_stats.groupby(
            ['recent_team', 'season']
        )['team_total_points'].transform(lambda x: x.expanding().mean().shift(1))

        # Handle Week 1 (Fill NaNs with the global average offense score)
        avg_offense_score = team_weekly_stats['team_total_points'].mean()
        team_weekly_stats['team_avg_points_scored'] = team_weekly_stats['team_avg_points_scored'].fillna(avg_offense_score)

        # Merge back into the main data
        # We match on 'recent_team' this time (not opponent)
        clean_data = pd.merge(
            clean_data,
            team_weekly_stats[['recent_team', 'season', 'week', 'team_avg_points_scored']],
            on=['recent_team', 'season', 'week'],
            how='left'
        )
        
        print("Finished Team Offense features.")

        
        ### CAREER BASELINE (Expanding Past Seasons Average) ###


        # We have to use the raw data for this calculation
        temp_data = self.raw_weekly_data.copy()

        # We need to fill NaNs here to prevent the filter from crashing
        temp_data['offense_snaps'].fillna(0, inplace=True)

        # Aggregate to the SEASON level

        # Filter out games they didn't actually play in
        played_games_data = temp_data[temp_data['offense_snaps'] > 0]

        # Use our grouping keys to map the data, and then get the average for every feature
        seasonal_stats = played_games_data.groupby(grouping_keys)[stat_columns].mean().reset_index()

        # Sort the data to ensure seasons are in order (2020, 2021, 2022...)
        seasonal_stats = seasonal_stats.sort_values(['player_id', 'season'])
        
        # Calculate "Weighted Career Baseline" (Seasonal EWMA)
        # We use a span of 3 seasons. This means the last 3 years matter most.
        ewm_baseline_cols = {stat: f'career_ewm_{stat}' for stat in stat_columns}

        seasonal_stats[list(ewm_baseline_cols.values())] = seasonal_stats.groupby(
            ['player_id'], group_keys=False
        )[stat_columns].apply(lambda x: x.ewm(span=3, adjust=False).mean().shift(1))

        
        # Calculate "Last Season Stats"
        last_season_cols = {stat: f'last_season_{stat}' for stat in stat_columns}

        seasonal_stats[list(last_season_cols.values())] = seasonal_stats.groupby(
            ['player_id'], group_keys=False
        )[stat_columns].shift(1)

        # Merge BOTH sets of new features
        # We need to collect all the new column names we just created
        new_features = list(ewm_baseline_cols.values()) + list(last_season_cols.values())
        
        merge_cols = grouping_keys + new_features
        clean_data = pd.merge(
            clean_data,
            seasonal_stats[merge_cols],
            on=grouping_keys,
            how='left'
        )

        # Now we can drop the raw offense_snaps we brought over
        if 'offense_snaps' in clean_data.columns:
             clean_data = clean_data.drop(columns=['offense_snaps'])

        print("Finished 'Career Baseline' features.")

        # Finally, set the class variable
        self.feature_engineered_data = clean_data
        print("--- Feature Engineering Complete ---")        

    def preprocess(self):
        """
        Performs any final preprocessing steps on self.features before training.
        """
        print("--- Preprocessing Data ---")

        # We're filling all missing numeric values with 0.        
        numeric_cols = self.feature_engineered_data.select_dtypes(include=[np.number]).columns
        self.feature_engineered_data[numeric_cols] = self.feature_engineered_data[numeric_cols].fillna(0)
        print("Filled all missing numeric values with 0.")
        
        # Encode Categorical Features
        try:
            self.feature_engineered_data = pd.get_dummies(
                self.feature_engineered_data, 
                columns=['position'], 
                dummy_na=False # We don't want a column for "NaN" positions
            )
            print("Successfully one-hot encoded categorical features.")

            # We replaced these string columns with numerical stats
            cols_to_drop = ['opponent_team', 'recent_team']
            
            # Only drop them if they actually exist
            existing_cols_to_drop = [c for c in cols_to_drop if c in self.feature_engineered_data.columns]
            
            if existing_cols_to_drop:
                self.feature_engineered_data.drop(columns=existing_cols_to_drop, inplace=True)
                print(f"Dropped raw string columns: {existing_cols_to_drop}")
            
        except Exception as e:
            print(f"Error during one-hot encoding: {e}")

    def split_data(self, test_season: int = 2024, burn_in_years: int = 1):
        """
        Splits the feature-engineered data (self.features, self.target)
        into training and testing sets.
        """
        print(f"--- Splitting Data (Test: {test_season}, Burn-in: {burn_in_years} yrs) ---")
        
        if self.feature_engineered_data is None:
            print("Error: No feature-engineered data to split. Run engineer_features() first.")
            return

        # Determine the start of valid training data
        min_season = self.feature_engineered_data['season'].min()
        train_start_season = min_season + burn_in_years
        
        print(f"Raw Data Start: {min_season}")
        print(f"Training Start: {train_start_season} (First {burn_in_years} years used for history only)")

        # Create the boolean masks
        # Test Set: The specific test season
        test_mask = self.feature_engineered_data['season'] == test_season
        # Train Set: From train_start_season up to (but not including) test_season
        train_mask = (self.feature_engineered_data['season'] >= train_start_season) & \
                     (self.feature_engineered_data['season'] < test_season)

        # Apply the masks to get our two datasets
        train_data = self.feature_engineered_data[train_mask]
        test_data = self.feature_engineered_data[test_mask]

        print(f"Training set shape: {train_data.shape}")
        print(f"Testing set shape: {test_data.shape}")

        # Define our target and non-feature columns
        target_column = 'fantasy_points_ppr'
        
        # These are all the columns that we DON'T want the model to learn from
        non_feature_cols = [
            'fantasy_points_ppr', 'fantasy_points', 'player_id', 'player_name', 
            'player_display_name', 'season', 'week', 'ewm_fantasy_points_ppr',
            'expanding_fantasy_points_ppr', 'career_ewm_fantasy_points_ppr', 
            'last_season_fantasy_points_ppr'
        ]
        
        # Create the final X/y splits
        self.training_features = train_data.drop(columns=non_feature_cols, errors='ignore')
        self.training_target = train_data.loc[:, target_column]
        
        self.testing_features = test_data.drop(columns=non_feature_cols, errors='ignore')
        self.testing_target = test_data.loc[:, target_column]

        print("Data successfully split.")

    def train_model(self):
        """
        Trains the machine learning model on the training data.
        """
        print("--- Training Model ---")
        
        # Initialize the XGBRegressor
        self.trained_model = xgb.XGBRegressor(
            n_estimators=1000,          # How many trees to build (We will use other criteria to stop it early)
            learning_rate=0.01,         # How much a new tree should correct the previous one
            early_stopping_rounds=10,   # Stop training if there is no improvement for 10 trees
            random_state=42,             # For reproducibility
            base_score=0.5
        )
        
        print("Model initialized. Starting training...")
        
        # Train the model
        self.trained_model.fit(
            self.training_features,
            self.training_target,
            eval_set=[(self.testing_features, self.testing_target)],
            verbose=False # Set this to True if you want to see the progress of every round
        )

        # Save model to file to help SHAP parse it
        model_filename = "xgb_model.json"
        self.trained_model.save_model(model_filename)
        print(f"Model saved to {model_filename}")
        
        print("--- Model Training Complete ---")
        print(f"Model stopped at iteration: {self.trained_model.best_iteration}")

    def load_saved_model(self, filename="xgb_model.json"):
        """
        Loads a pre-trained XGBoost model from a JSON file.
        This allows you to skip training and go straight to prediction.
        """
        print(f"--- Loading Saved Model from {filename} ---")
        
        # We must initialize the architecture first
        self.trained_model = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.01, 
            early_stopping_rounds=10, 
            random_state=42
        )
        
        try:
            # Load the weights/trees from the file
            self.trained_model.load_model(filename)
            print("Model successfully loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Did you run train_model() at least once to create the file?")

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set (self.testing_features).
        """
        print("--- Evaluating Model ---")
        
        # Make predictions on the test data (2024 season)
        predictions = self.trained_model.predict(self.testing_features)
        
        # Calculate the regression metrics from class
        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(self.testing_target, predictions)
        
        # RMSE (Root Mean Squared Error)
        mse = mean_squared_error(self.testing_target, predictions)
        rmse = np.sqrt(mse)

        # Functional Accuracy
        # We define a 'Hit' as being within 5.0 fantasy points of the truth.
        tolerance = 5.0
        errors = np.abs(self.testing_target - predictions)
        within_tolerance = (errors <= tolerance)
        accuracy_within_5 = within_tolerance.mean()
        
        # Calculate the natural standard deviation of the ACTUAL data.
        # This represents the "randomness" of the game.
        natural_std_dev = self.testing_target.std()
        
        # Did we predict closer than the randomness of the game?
        within_noise = (errors <= natural_std_dev)
        accuracy_within_noise = within_noise.mean()

        # Print the results
        print("--- Model Evaluation Results ---")
        print(f"  Mean Absolute Error (MAE):  {mae:.4f} (fantasy points)")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} (fantasy points)")
        print("  Custom Accuracy Metrics:")
        print(f"  Natural Noise of Test Set (StdDev): {natural_std_dev:.2f} pts")
        print(f"  Accuracy (within Natural Noise): {accuracy_within_noise:.2%}")
        print(f"  Accuracy (within {tolerance} pts):      {accuracy_within_5:.2%}")
        print("----------------------------------")


    def explain_model(self):
        """
        Uses the SHAP library to explain the model's predictions.
        """
        print("--- Explaining Model (Calculating SHAP values) ---")
        
        if self.trained_model is None:
            print("Error: Model not trained. Please run train_model() first.")
            return

        # Initialize the SHAP Explainer
        explainer = shap.TreeExplainer(
            model=self.trained_model, 
            feature_perturbation="tree_path_dependent"
        )

        shap_values = explainer(self.training_features)
        
        # Store the results in our class
        self.model_explainer = explainer
        self.feature_importance_values = shap_values #.values
        
        print("--- SHAP Value Calculation Complete ---")

    def plot_feature_importance(self, num_features: int = 20):
        """
        Generates and saves a SHAP summary plot (bar chart) to show
        global feature importance.

        Args:
            num_features (int): How many top features to show on the plots.
        """
        print("--- Plotting Feature Importance ---")

        MY_GREEN = '#30df7a'
        TEXT_COLOR = '#2c3e50'
        BG_COLOR = '#f8f9fa'
        
        if self.feature_importance_values is None:
            print("Error: SHAP values not calculated. Please run explain_model() first.")
            return

        # Ensure the 'plots/' directory exists
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            print(f"Created directory: {plot_dir}")

        # Apply Global Style Settings for this plot
        plt.rcParams['text.color'] = TEXT_COLOR
        plt.rcParams['axes.labelcolor'] = TEXT_COLOR
        plt.rcParams['xtick.color'] = TEXT_COLOR
        plt.rcParams['ytick.color'] = TEXT_COLOR
        plt.rcParams['figure.facecolor'] = BG_COLOR
        plt.rcParams['axes.facecolor'] = BG_COLOR

        #### GLOBAL FEATURE IMPORTANCE ####
        # This plot shows the most important features across ALL positions.
        print("Generating global feature importance plot...")
        
        # Increase figure size to prevent title clipping
        fig = plt.figure(figsize=(14, 10))
        
        shap.summary_plot(
            self.feature_importance_values, 
            self.training_features, 
            plot_type="bar", 
            max_display=num_features, 
            show=False,
            color=MY_GREEN
        )
        
        # Customize the text/labels AFTER plotting

        # This forces extra whitespace
        plt.subplots_adjust(left=0.45, right=0.95, top=0.9, bottom=0.1)
        
        ax = plt.gca()
        # ax.set_title("Global Feature Importance", fontsize=22, fontweight='bold', pad=20, color=TEXT_COLOR)
        ax.set_xlabel("Relative Importance", fontsize=16, color=TEXT_COLOR, labelpad=10)

        # Use Figure-level title to avoid clipping
        fig.suptitle("Global Feature Importance", fontsize=22, fontweight='bold', y=0.95, color=TEXT_COLOR)
        
        # Fix tick label sizes
        ax.tick_params(axis='both', labelsize=12)

        # Save with 'bbox_inches' to prevent text cutoff
        save_path = os.path.join(plot_dir, 'global_feature_importance.png')
        plt.savefig(save_path, facecolor=BG_COLOR)
        print(f"Saved global plot to: {save_path}")
        plt.close()

        #### POSITION-SPECIFIC PLOTS ####
        print("Generating position-specific feature importance plots...")
        
        positions = ['QB', 'RB', 'WR', 'TE']
        
        # Get a list of all position columns
        all_pos_cols = [
            f'position_{p}' for p in positions 
            if f'position_{p}' in self.training_features.columns
        ]

        for pos in positions:
            pos_col_name = f'position_{pos}'
            
            # Check if this position exists in our data
            if pos_col_name not in self.training_features.columns:
                print(f"Skipping plot for {pos}: no data found.")
                continue

            # Create a Boolean Mask (True/False for every row)
            pos_mask = self.training_features[pos_col_name] == 1

            if not pos_mask.any():
                print(f"Skipping plot for {pos}: no data found.")
                continue

            # Use the mask to filter the SHAP values (Numpy Array)
            pos_shap_values = self.feature_importance_values.values[pos_mask]

            # Use the mask to filter the Features (DataFrame)
            pos_features = self.training_features[pos_mask]
            
            # Remove position columns from the plot.
            pos_features_cleaned = pos_features.drop(columns=all_pos_cols, errors='ignore')
            
            # We also need to get the SHAP values for *only* the cleaned features
            cleaned_col_indices = [
                pos_features.columns.get_loc(c) for c in pos_features_cleaned.columns
            ]
            # Create a new SHAP values object with just these columns
            pos_shap_values_cleaned = pos_shap_values[:, cleaned_col_indices]
            
            # Now, generate the plot for this position
            fig = plt.figure(figsize=(14, 10)) # Consistent size
            
            shap.summary_plot(
                pos_shap_values_cleaned, 
                pos_features_cleaned, 
                plot_type="bar", 
                max_display=num_features, 
                show=False,
                color=MY_GREEN
            )

            # Force Margins
            plt.subplots_adjust(left=0.45, right=0.95, top=0.9, bottom=0.1)

            # Custom Titles & Labels
            ax = plt.gca()
            # ax.set_title(f"Feature Importance: {pos}", fontsize=18, fontweight='bold', pad=20, color=TEXT_COLOR)
            ax.set_xlabel("Relative Importance", fontsize=14, color=TEXT_COLOR)
            fig.suptitle(f"Feature Importance: {pos}", fontsize=22, fontweight='bold', y=0.95, color=TEXT_COLOR)
            
            ax.tick_params(axis='both', labelsize=12)

            save_path = os.path.join(plot_dir, f'feature_importance_{pos}.png')
            plt.savefig(save_path, facecolor=BG_COLOR)
            print(f"Saved {pos} plot to: {save_path}")
            plt.close()

        print("--- Plotting Complete ---")

    def run_pipeline(self, do_eda=True, test_season: int = 2024, burn_in_years: int = 1):
        """
        Runs the pipeline in order.
        
        Args:
            do_eda (bool): Whether to run and save EDA plots.
            test_season (int): The season to reserve for testing.
            burn_in_years (int): Number of early years to exclude from training 
                                 (used for historical stat calculation only).
        """
        print("Starting data pipeline...")
        self.load_data()

        if(do_eda):
            print("Performing Exploratory Data Analysis (EDA)...")
            self.perform_eda()
        
        print("Engineering features...")
        self.engineer_features()
        
        print("Preprocessing data...")
        self.preprocess()

        display(self.feature_engineered_data)
        
        # Try to save our polised data to a csv file
        try:
            debug_path = 'data/final_model_ready_data.csv'
            self.feature_engineered_data.to_csv(debug_path, index=False)
            print(f"Saved model-ready data for inspection to: {debug_path}")
        except Exception as e:
            print(f"Error saving debug CSV: {e}")
        
        print("Splitting data...")
        self.split_data(test_season=test_season, burn_in_years=burn_in_years)
        
        print("Training model...")
        self.train_model()    
        
        print("Evaluating model...")
        self.evaluate_model()
        
        print("Explaining model...")
        self.explain_model()
        
        print("Plotting feature importance...")
        self.plot_feature_importance()
        
        print("Pipeline complete.")