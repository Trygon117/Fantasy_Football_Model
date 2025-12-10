# Fantasy Football Prediction Pipeline

    This project is a machine learning pipeline designed to generate highly accurate weekly fantasy football point projections for Skill Positions (QB, RB, WR, TE). The model uses XGBoost and advanced time-series feature engineering to outperform baseline models.

## What This Project Does

    The core function of this pipeline is to ingest 10 years of NFL play-by-play data, analyze weekly trends, and predict player performance based on contextual features (e.g., opponent defensive weakness, recent form).

    The output is a ranked CSV file ({year}_week_{week}_projections.csv) that provides actionable start/sit recommendations for the upcoming week.
    
## Setup and Installation

### Install Dependencies: All necessary Python libraries can be installed using the requirements.txt file.

    pip install -r requirements.txt

## Project Structure & How to Run

### The entire pipeline is designed to be executed step-by-step from the main.ipynb Jupyter Notebook, which orchestrates the process and provides visualizations.
    Core Files:
        File Name	         Purpose
        main.ipynb	         The Entry Point. Executes the full pipeline: Data download, feature engineering, model training, and final prediction output.
        data_processing.py	 Handles raw data ingestion and initial cleaning (via nflreadpy).
        fantasy_model.py	 Defines the FantasyModel class, which manages feature creation and the XGBoost training process.
        prediction.py	     Contains the function to generate projections for future, unplayed games.

### To run the pipeline and generate predictions for the next week, simply open and execute the cells in main.ipynb sequentially.

    Data: Ingests and cleans 10 years of data.

    Train: Trains the XGBoost model and prints validation metrics.

    Predict: Outputs the {year}_week_{week}_projections.csv file with your final rankings.