import json
import logging
import os
import threading
import time as t
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from django.http import HttpResponse
from joblib import load

pd.options.mode.chained_assignment = None


class Weather():
    """
    Responsible for gathering weather data from DarkSky API.
    cache( ) method caches weather data dictionary
    update_rain_temp( ) method indexes correct weather period from cache

    """

    def __init__(self):

        self.current_temperature = None
        self.current_rainfall = None
        self.temp = None
        self.rain = None
        self.weather_forecast_json = None
        self.logger = logging.getLogger(__name__)

    def cache(self):

        """
        Requests a JSON object from the DarkSky API.
        Updates the current temperature and rainfall.
        """

        api = ('https://api.darksky.net/forecast/98cdd61d77bab4d8d739f78b33'
               'e06c30/53.3498,-6.2603?units=si')

        current_weather_data = requests.get(api)

        if current_weather_data.status_code == 200:
            current_weather_data = json.loads(current_weather_data.text)

            self.weather_forecast_json = current_weather_data
            self.current_temperature = self.weather_forecast_json \
                ['currently']['temperature']
            self.current_rainfall = self.weather_forecast_json \
                ['currently']['precipIntensity']

        else:
            self.logger.error('Darksky API call failed.')

        threading.Timer(1200.0, self.cache).start()

    def update_rain_temp(self, day_of_week, departure_time_seconds):

        """
        Takes the day of the week and the time of day to index the
        weather dictionary and updates the temperature and rainfall values
        for that time.
        """

        current_time = t.time()
        today = datetime.today().weekday()

        if (departure_time_seconds < (current_time + 3600) \
                and day_of_week == today):

            self.temp = self.current_temperature
            self.rain = self.current_rainfall

        elif (day_of_week == today):
            for i in range(24):
                if (departure_time_seconds > self.weather_forecast_json \
                        ["hourly"]["data"][i]["time"] and departure_time_seconds \
                        < self.weather_forecast_json["hourly"]["data"][i + 1]["time"]):

                    self.temp = self.weather_forecast_json \
                        ['hourly']['data'][i]['temperature']

                    self.rain = self.weather_forecast_json['hourly'] \
                        ['data'][i]['precipIntensity']
                    break
                else:
                    continue
        else:
            day_difference = int((departure_time_seconds - current_time) / 86400)

            self.temp = (self.weather_forecast_json['daily']['data'] \
                             [day_difference]['temperatureMax'] + \
                         self.weather_forecast_json['daily']['data'] \
                             [day_difference]['temperatureMin']) / 2

            self.rain = self.weather_forecast_json['daily'] \
                ['data'][day_difference]['precipIntensity']


class Time():
    """
    Class that converts time elements from Javascript to Python formats.
    """

    def __init__(self):

        self.date = None
        self.day_of_week = None

    def find_date(self, departure_time_seconds):

        """
        Creates string attribute: date of prediction.
        """

        self.date = datetime.fromtimestamp(departure_time_seconds).strftime('%d/%m/%Y')

    def check_day(self, day_of_week):

        """
        Changes the day encoding from Javascript to Python format.
        Python: 0=Monday, 6=Sunday
        """

        day_of_week -= 1
        if (day_of_week == -1):
            self.day_of_week = 6
        else:
            self.day_of_week = day_of_week


class FormatInput():

    def __init__(self, route_id, stop_id, planned_arrival, rain, temp,
                 day_of_week, month, t_minus_1118, datestring, lat_lon_dict,
                 cluster_dict):

        """
        Constructor
        — indexes parameters passed
        — loads a dictionary containing latitude and longitude coordinates
        for each bus stop
        — loads a dictionary that assigns bus stops to a geographical
        cluster (area)

        """

        self.line = route_id
        self.stop_id = stop_id
        self.planned_arrival = planned_arrival
        self.rain = rain
        self.temp = temp
        self.day_of_week = day_of_week
        self.month = month
        self.t_minus_1118 = t_minus_1118
        self.date = datestring

        # latitude and longitude dictionary

        json_obj = open(lat_lon_dict)
        json_obj = json_obj.read()
        stops_latlon = json.loads(json_obj)

        self.stops_latlon = {int(k): stops_latlon[k] for k in stops_latlon.keys()}
        self.stops_lat = {k: v[0] for k, v in stops_latlon.items()}
        self.stops_lon = {k: v[1] for k, v in stops_latlon.items()}

        # bus cluster dictionary

        json_obj = open(cluster_dict)
        json_obj = json_obj.read()
        cluster_map_original = json.loads(json_obj)

        self.cluster_map = {}

        for k, v in cluster_map_original.items():
            for item in v:
                self.cluster_map[item] = k

    def haversine(lat2, lon2):

        """
        Helper function that calculates the great-circle
        distance between a pair of geographical coordinates.

        Implementation taken from https://stackoverflow.com/questions/29545704/
        fast-haversine-approximation-python-pandas.

        Returns:
            [float] -- [the distance]
        """

        lat1 = 53.342998628
        lon1 = -6.256165642

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c

        return km

    def map_stop_to_cluster(cluster_map, stop_id):

        """
        Helper function that assigns a bus stop number to a geographical
        cluster. References a dictionary loaded in the
        class constructor to do so.

        Returns:
            [int] -- [a categorical integer value, representing
            a cluster]
        """

        stop_id = int(stop_id)

        while stop_id not in cluster_map.keys():
            if stop_id < 7692:
                stop_id += 1
            else:
                while stop_id not in cluster_map.keys():
                    stop_id -= 1

        return cluster_map[stop_id]

    def add_holiday(date):

        """
        Helper function assigns a value based on whether the date for prediction
        is a bank, national, and / or primary school holiday.

        Returns:
            [int] -- [0: not a holiday; 1: is a holiday]
        """

        holiday_dates = ['09/08/2019', '10/08/2019', '11/08/2019', '12/08/2019',
                         '13/08/2019', '14/08/2019', '15/08/2019', '16/08/2019', '17/08/2019',
                         '18/08/2019', '19/08/2019', '20/08/2019', '21/08/2019', '22/08/2019',
                         '23/08/2019', '24/08/2019', '25/08/2019', '26/08/2019', '27/08/2019',
                         '28/08/2019', '29/08/2019', '30/08/2019', '31/08/2019', '28/10/2019',
                         '29/10/2019', '30/10/2019', '31/10/2019', '20/12/2019', '21/12/2019',
                         '22/12/2019', '23/12/2019', '24/12/2019', '25/12/2019', '26/12/2019',
                         '27/12/2019', '28/12/2019', '29/12/2019', '30/12/2019', '31/12/2019',
                         '01/01/2020', '02/01/2020', '03/01/2020', '04/01/2020', '05/01/2020',
                         '17/02/2020', '18/02/2020', '19/02/2020', '20/02/2020', '21/02/2020',
                         '03/04/2020', '04/04/2020', '05/04/2020', '06/04/2020', '07/04/2020',
                         '08/04/2020', '09/04/2020', '10/04/2020', '11/04/2020', '12/04/2020',
                         '13/04/2020', '14/04/2020', '15/04/2020', '16/04/2020', '17/04/2020',
                         '18/04/2020', '19/04/2020', '20/04/2020', '28/06/2020', '29/06/2020',
                         '30/06/2020', '31/06/2020', '01/07/2020', '02/07/2020', '03/07/2020',
                         '04/07/2020', '05/07/2020', '06/07/2020', '07/07/2020', '08/07/2020',
                         '09/07/2020', '10/07/2020', '11/07/2020', '12/07/2020', '13/07/2020',
                         '14/07/2020', '15/07/2020', '16/07/2020', '17/07/2020', '18/07/2020',
                         '19/07/2020', '20/07/2020', '21/07/2020', '22/07/2020', '23/07/2020',
                         '24/07/2020', '25/07/2020', '26/07/2020', '27/07/2020', '28/07/2020',
                         '29/07/2020', '30/07/2020', '31/07/2020', '01/08/2020', '02/08/2020',
                         '03/08/2020', '04/08/2020', '05/08/2020', '06/08/2020', '07/08/2020',
                         '08/08/2020', '09/08/2020', '10/08/2020', '11/08/2020', '12/08/2020',
                         '13/08/2020', '14/08/2020', '15/08/2020', '16/08/2020', '17/08/2020',
                         '18/08/2020', '19/08/2020', '20/08/2020', '21/8/2020', '22/08/2020',
                         '23/08/2020', '24/08/2020', '25/08/2020', '26/08/2020', '27/08/2020',
                         '28/08/2020', '29/08/2020', '30/08/2020', '31/08/2020']

        if date in holiday_dates:
            return 1
        else:
            return 0

    def derive_features(self):

        """
        Instance method that applies the haversine, cluster, and holiday
        functions to input parameters.

        Results are saved as instance attributes.
        """

        temp = int(self.stop_id)

        while temp not in self.stops_latlon.keys():
            if temp < 7692:
                temp += 1
            else:
                while temp not in self.stops_latlon.keys():
                    temp -= 1

        self.latitude = self.stops_latlon[temp][0]
        self.longitude = self.stops_latlon[temp][1]

        self.distance_centre = FormatInput.haversine(self.latitude, self.longitude)

        self.cluster = FormatInput.map_stop_to_cluster(self.cluster_map, self.stop_id)

        self.holiday = FormatInput.add_holiday(self.date)

    def label_encode(self, directory):

        """
        Instance Method that applies the appropriate and required label encoding
        values to categorical inputs.

        The encoding dictionary was generated in model training, line per line.

        Three prediction input parameters require label encoding.
        """

        dataframe = 'df_' + str(self.line) + '/'

        line_encodings_path = os.path.join(directory, dataframe)

        month_source = line_encodings_path + 'MONTH_encode_map.json'
        stop_id_source = line_encodings_path + 'STOPPOINTID_encode_map.json'
        cluster_source = line_encodings_path + 'cluster_encode_map.json'

        features = [self.month, self.stop_id, self.cluster]
        dict_sources = [month_source, stop_id_source, cluster_source]

        for i in range(len(dict_sources)):
            json_obj = open(dict_sources[i])
            json_obj = json_obj.read()
            dictionary = json.loads(json_obj)
            key = features[i]
            features[i] = dictionary[str(key)]

    def create_vector(self):

        """
        Instance method that create a prediction model input
        vector from user parameters and parameters generated in
        this class.
        """

        series = {'PLANNEDTIME_ARR': [int(self.planned_arrival)],
                'rain': [float(self.rain)],
                'temp': [float(self.temp)],
                'distance_centre': [float(self.distance_centre)],
                'day_into_year': [int(self.t_minus_1118)],
                'STOPPOINTID_label': [int(self.stop_id)],
                'DAYOFWEEK_label': [int(self.day_of_week)],
                'MONTH_label': [int(self.month)],
                'holiday_label': [int(self.holiday)],
                'cluster_label': [int(self.cluster)]
                }

        self.vector = pd.DataFrame(series)


class Predict():
    """
    A class that returns a predicted time of arrival at a
    bus stop, in seconds past midnight.
    """

    def predict(self, models_dir, line_number, vector):
        """
        Instance method. For a specified line number and
        input vector, generates a prediction.

        Loads the appropriate model for a line from memory.
        """

        modelname = 'df_' + str(line_number) + '.joblib'

        filepath = os.path.join(models_dir, modelname)

        model = load(filepath)

        names = model.get_booster().feature_names

        result = model.predict(vector[names].iloc[[-1]])
        return int(result)


def predict(request, route_id, stop_id, planned_arrival, day_of_week, month,
            departure_time_seconds, t_minus_1118):
    """

	This is the main method.

    Executes the program (is __main__)
    Formats input parameters, generates a preduction.
    Returns the prediction.

    INPUT: [int] route_id, [int] stop_id, [int] planned_arrival,
    [int] day_of_week (Monday=0, Sunday=6), [int] month, [int] t_minus_1118
    (days since Jan1, 2018), [str] date_string (format: '00/00/0000')
    """

    stop_lat_lon_dict = '/home/student/analytics/ml_dictionaries/db_stop_latlon.json'

    cluster_dict = '/home/student/analytics/ml_dictionaries/stops_to_cluster_300m_radius.json'

    labels_dir = '/home/student/analytics/models_label_encodings/'

    models_dir = '/home/student/analytics/models/'

    # this call updates the weather attributes to the current time and / or day
    # implement front-end check to queries to JSON response out of range

    time = Time()
    time.find_date(departure_time_seconds)
    time.check_day(day_of_week)

    print(time.day_of_week, departure_time_seconds)
    weather.update_rain_temp(time.day_of_week, departure_time_seconds)

    # this instantiates our formatter object with prediction inputs

    formatter = FormatInput(route_id,
                            stop_id,
                            planned_arrival,
                            weather.rain,
                            weather.temp,
                            time.day_of_week,
                            month,
                            t_minus_1118,
                            time.date,
                            stop_lat_lon_dict,
                            cluster_dict)

    # this adds and transforms features in preparation for prediction

    formatter.derive_features()

    # this label encodes the appropriate categorical variables

    formatter.label_encode(labels_dir)

    # this creates an input vector attribute to pass to the predictor
    formatter.create_vector()

    # instantiate a predictor object

    predictor = Predict()

    # generate a prediction for a line, given a vector which is passed

    result = predictor.predict(models_dir, formatter.line, formatter.vector)

    # return the prediction

    return HttpResponse(result)


# Django will the following

weather = Weather()
weather.cache()
