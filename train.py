import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import os

def main(args):
    # Load dataset
    df = pd.read_csv(args.training_data)
    
    # Simple Preprocessing
    # Features: hour, temperature, humidity, windspeed, workingday
    X = df[['hr', 'temp', 'hum', 'windspeed', 'workingday']]
    y = df['cnt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLflow tracking
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Log results
        print(f"Model MAE: {mae}")
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        
        # Save model for Azure deployment
        mlflow.sklearn.log_model(model, "bike_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, help="Path to training data")
    args = parser.parse_args()
    main(args)
