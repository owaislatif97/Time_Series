import sys
import os
# Ensure the parent directory is included in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecast.A_data_loader import DataLoader
from src.forecast.B_preprocessing import DataPreprocessor
from src.forecast.C_models import ModelTrainer
from src.forecast.D_predictor import Predictor
from src.forecast.E_evaluation import Evaluator
from src.forecast.A_user_input import (
    get_user_input,
    get_location_choice,
    display_column_info
)

def interactive_prediction_mode(predictor, df):
    """
    Run interactive prediction mode where user can input parameters
    """
    # First, display info about the features to help the user
    display_column_info(df)

    while True:
        parameters = get_user_input()

        try:
            predictions = predictor.predict_all_from_parameters(parameters)

            print("\n=== One-Hour Ahead Power Predictions ===")
            for model_name, prediction in predictions.items():
                print(f"{model_name.upper()} Model: {prediction:.4f}")
        except Exception as error:
            print(f"Error making prediction: {str(error)}")

        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            break

def main():
    location_index = get_location_choice()

    # Step 1: Load data
    print(f"Step 1: Loading data from Location {location_index}...")
    loader = DataLoader()
    df = loader.load_data(location_index)

    data_info = loader.get_data_info()
    print(f"Total dataset size: {data_info['rows']} rows")
    print(f"Dataset time range: {data_info['time_range'][0]} to {data_info['time_range'][1]}")

    # Step 2: Preprocess
    print("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, time_test = preprocessor.preprocess(df)
    print("✅ Preprocessing complete!")

    last_n_points = 500
    print(f"\nWe'll visualize the last {last_n_points} data points")

    # Step 3: Persistence model
    print("\n🔮 TASK 1: One-hour ahead prediction with Persistence Model...")
    predictor = Predictor()
    y_true_persist, y_pred_persist, time_persist = predictor.predict_persistence(df)

    evaluator = Evaluator()
    evaluator.evaluate_model(y_true_persist, y_pred_persist, model_name="persistence")
    evaluator.print_metrics("persistence")

    # Step 4: ML models
    print("\n🔮 TASK 2: One-hour ahead prediction with ML Models...")
    X_train_1h, X_test_1h, y_train_1h, y_test_1h, time_test_1h = preprocessor.prepare_features_for_one_hour_ahead()

    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train_1h, y_train_1h)

    predictor = Predictor(models)
    predictor.set_scaler(preprocessor.scaler, preprocessor.X_raw.columns.tolist())
    predictions = predictor.predict_with_models(X_test_1h)
    predictions['persistence'] = y_pred_persist[-len(y_test_1h):]

    for name, y_pred in predictions.items():
        evaluator.evaluate_model(y_test_1h, y_pred, model_name=name)

    evaluator.print_metrics()

    # Step 5: Plot predictions
    print("\n🔮 TASK 3: Plotting one-hour ahead predictions for Last 500 points...")
    evaluator.plot_predictions(time_test_1h, y_test_1h, predictions, last_n_points)

    print("\n✅ All tasks completed successfully!")

    # Step 6: Interactive mode
    print("\n🔮 TASK 4: Interactive one-hour ahead predictions...")
    interactive_mode = input("\nWould you like to enter interactive prediction mode? (y/n): ")
    if interactive_mode.lower() == 'y':
        interactive_prediction_mode(predictor, df)

if __name__ == "__main__":
    main()
