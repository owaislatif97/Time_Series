"""
user_input.py
Handles user input and display utilities for wind power forecasting project.
"""
import pandas as pd

def display_column_info(df: pd.DataFrame) -> None:
    """
    Display information about the features in the dataset, excluding Time and Power.

    Parameters:
        df (pd.DataFrame): The dataset.
    """
    print("\nDataset features:")
    for col in df.columns:
        if col not in ['Time', 'Power']:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"- {col}: range [{min_val:.2f} to {max_val:.2f}], mean: {mean_val:.2f}")

def get_user_input() -> dict:
    """
    Prompt the user to input weather parameters for prediction.

    Returns:
        dict: User-input weather parameters.
    """
    print("\n=== Enter parameters for one-hour ahead prediction ===")
    parameters = {
        'temperature': float(input("Temperature at 2m (\u00b0C): ")),
        'humidity': float(input("Relative Humidity at 2m (%): ")),
        'dewpoint': float(input("Dewpoint at 2m (\u00b0C): ")),
        'windspeed_10m': float(input("Wind Speed at 10m (m/s): ")),
        'windspeed_100m': float(input("Wind Speed at 100m (m/s): ")),
        'winddirection_10m': float(input("Wind Direction at 10m (degrees): ")),
        'winddirection_100m': float(input("Wind Direction at 100m (degrees): ")),
        'windgusts': float(input("Wind Gusts at 10m (m/s): "))
    }

    current_power = input("\nCurrent Power (optional, for persistence model): ")
    if current_power.strip():
        parameters['current_power'] = float(current_power)

    return parameters

def get_location_choice() -> int:
    """
    Prompt user to select a location index.

    Returns:
        int: Location index selected by the user.
    """
    while True:
        print("\n=== Choose a location for wind data analysis ===")
        print("1: Location 1")
        print("2: Location 2")
        print("3: Location 3")
        print("4: Location 4")

        choice = input("Enter your choice (1-4): ")
        try:
            location_index = int(choice)
            if 1 <= location_index <= 4:
                return location_index
            print("Error: Please enter a number between 1 and 4.")
        except ValueError:
            print("Error: Please enter a valid number.")
