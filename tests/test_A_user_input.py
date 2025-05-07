import unittest
from unittest.mock import patch, MagicMock
import io
import pandas as pd
import sys
sys.path.append('.')  # Adds the project directory to the path
from src.forecast.A_user_input import get_user_input, get_location_choice, display_column_info

class TestUserInput(unittest.TestCase):

    def setUp(self):
        """Set up test data that will be used by multiple tests."""
        # Create a sample DataFrame for testing
        self.test_df = pd.DataFrame({
            'Time': pd.date_range(start='2023-01-01', periods=5, freq='h'),
            'Power': [100, 200, 150, 300, 250],
            'temperature': [20.5, 21.2, 19.8, 22.5, 23.1],
            'humidity': [65.2, 70.5, 62.8, 68.3, 59.7],
            'windspeed_10m': [5.2, 6.8, 4.5, 7.2, 8.1]
        })

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_display_column_info(self, mock_stdout):
        """Test that display_column_info displays the correct information."""
        display_column_info(self.test_df)
        output = mock_stdout.getvalue()

        # Verify the output contains expected elements
        self.assertIn("Dataset features:", output)
        self.assertIn("temperature:", output)
        self.assertIn("humidity:", output)
        self.assertIn("windspeed_10m:", output)

        # Verify the output contains the correct range and mean values
        self.assertIn("temperature: range [19.80 to 23.10], mean: 21.42", output)
        self.assertIn("humidity: range [59.70 to 70.50], mean: 65.30", output)
        self.assertIn("windspeed_10m: range [4.50 to 8.10], mean: 6.36", output)

        # Verify that Time and Power are excluded
        self.assertNotIn("Time: range", output)
        self.assertNotIn("Power: range", output)

    @patch('builtins.input')
    def test_get_user_input_without_current_power(self, mock_input):
        """Test get_user_input function without providing current power."""
        # Mock input values
        mock_inputs = [
            "22.5",  # Temperature
            "68.3",  # Humidity
            "15.6",  # Dewpoint
            "6.2",   # Wind Speed at 10m
            "9.8",   # Wind Speed at 100m
            "180",   # Wind Direction at 10m
            "190",   # Wind Direction at 100m
            "7.5",   # Wind Gusts
            ""       # Current Power (empty)
        ]
        mock_input.side_effect = mock_inputs

        result = get_user_input()

        # Verify the result contains the expected parameters
        expected_result = {
            'temperature': 22.5,
            'humidity': 68.3,
            'dewpoint': 15.6,
            'windspeed_10m': 6.2,
            'windspeed_100m': 9.8,
            'winddirection_10m': 180.0,
            'winddirection_100m': 190.0,
            'windgusts': 7.5
        }
        self.assertEqual(result, expected_result)

        # Verify 'current_power' is not included
        self.assertNotIn('current_power', result)

    @patch('builtins.input')
    def test_get_user_input_with_current_power(self, mock_input):
        """Test get_user_input function with current power provided."""
        # Mock input values
        mock_inputs = [
            "22.5",  # Temperature
            "68.3",  # Humidity
            "15.6",  # Dewpoint
            "6.2",   # Wind Speed at 10m
            "9.8",   # Wind Speed at 100m
            "180",   # Wind Direction at 10m
            "190",   # Wind Direction at 100m
            "7.5",   # Wind Gusts
            "250"    # Current Power
        ]
        mock_input.side_effect = mock_inputs

        result = get_user_input()

        # Verify the result includes current_power
        expected_result = {
            'temperature': 22.5,
            'humidity': 68.3,
            'dewpoint': 15.6,
            'windspeed_10m': 6.2,
            'windspeed_100m': 9.8,
            'winddirection_10m': 180.0,
            'winddirection_100m': 190.0,
            'windgusts': 7.5,
            'current_power': 250.0
        }
        self.assertEqual(result, expected_result)

    @patch('builtins.input')
    def test_get_location_choice_valid_input(self, mock_input):
        """Test get_location_choice with valid input."""
        # Test with valid location choices
        for location in range(1, 5):
            mock_input.return_value = str(location)
            result = get_location_choice()
            self.assertEqual(result, location)

    @patch('builtins.input')
    @patch('builtins.print')
    def test_get_location_choice_invalid_then_valid(self, mock_print, mock_input):
        """Test get_location_choice with invalid input followed by valid input."""
        # Mock first an invalid input, then a valid input
        mock_input.side_effect = ["0", "5", "abc", "2"]

        result = get_location_choice()

        # Verify the final result is correct
        self.assertEqual(result, 2)

        # Verify error messages were displayed
        mock_print.assert_any_call("Error: Please enter a number between 1 and 4.")
        mock_print.assert_any_call("Error: Please enter a valid number.")

    @patch('builtins.input', side_effect=["22.5", "invalid", "68.3", "15.6", "6.2", "9.8", "180", "190", "7.5", ""])
    @patch('builtins.print')
    def test_get_user_input_with_invalid_input(self, mock_print, mock_input):
        """Test get_user_input with invalid input."""
        # This should raise a ValueError when encountering "invalid" for humidity
        with self.assertRaises(ValueError):
            get_user_input()


if __name__ == '__main__':
    unittest.main()