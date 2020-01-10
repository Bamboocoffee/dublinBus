from datetime import datetime

from django.test import TestCase


class TestCase(TestCase):

    def forecast_temperature_returned(self):
        future_time = time.time() + 20000
        current_date_time = datetime.now()
        day_of_week = current_date_time.weekday()
        rainfall, temperature = precipitation_temperature(day_of_week,50102,future_time)
        self.assertIsInstance(temperature, float)

    def current_temperature_information(self):
        current_temperature = current_temperature
        self.assertIsNotNone(current_temperature)

    def current_rainfall_information(self):
        current_rainfall = current_rainfall
        self.assertIsNotNone(current_rainfall)

    def weather_forecast_json_information(self):
        weather_forecast_jsone = weather_forecast_json
        self.assertIsNotNone(weather_forecast_json)

    def current_rainfall_returned(self):
        current_rainfall = current_rainfall
        current_time = time.time()
        current_date_time = datetime.now()
        day_of_week = current_date_time.weekday()
        rainfall, temperature = precipitation_temperature(day_of_week,50102,current_time)
        self.assertEqual(rainfall == current_rainfall)

    def current_temerature_returned(self):
        current_temperature = current_temperature
        current_time = time.time()
        current_date_time = datetime.now()
        day_of_week = current_date_time.weekday()
        rainfall, temperature = precipitation_temperature(day_of_week,50102,current_time)
        self.assertEqual(temperature == current_temperature)

    def forecast_rainfall_returned(self):
        future_time = time.time() + 20000
        current_date_time = datetime.now()
        day_of_week = current_date_time.weekday()
        rainfall, temperature = precipitation_temperature(day_of_week,50102,future_time)
        self.assertIsInstance(rainfall, float)


if __name__ == "__main__":
    unittest.main(argv=['ignored', '-v'], exit=False)
