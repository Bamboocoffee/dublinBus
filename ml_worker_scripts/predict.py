import os
import sys
import json
import numpy as np
import pandas as pd
from joblib import load
from math import sin, cos
from sklearn.preprocessing import LabelEncoder




class FormatInput():

   """
   A class that creates derivative features from input data
   and
   label-encodes the categorical variables
   and
   creators a valuesa vector for use in prediction.
   """


   def __init__(self, parameters, lat_lon_dict, cluster_dict):

      """
      Constructor
       — indexes parameters passed
       — loads a dictionary containing latitude and longitude coordinates
       for each bus stop
       — loads a dictionary that assigns bus stops to a geographical
       cluster (area)

      """

      self.line = parameters[0] # str # df_17A
      self.stop_id = parameters[1] # int
      self.planned_arrival = parameters[2] # int, second after midnight
      self.rain = parameters[3] # float
      self.temp = parameters[4] # flaot
      self.day_of_week = parameters[5] # int
      self.month = parameters[6] # int
      self.t_minus_1118 = parameters[7] # int
      self.date = parameters[8] # str # '9/8/19'


      # latitude and longitude dictionary

      json_obj = open(lat_lon_dict)
      json_obj = json_obj.read()
      stops_latlon = json.loads(json_obj)

      self.stops_latlon = {int(k): stops_latlon[k] for k in stops_latlon.keys()}
      self.stops_lat = {k:v[0] for k, v in stops_latlon.items()}
      self.stops_lon = {k:v[1] for k, v in stops_latlon.items()}

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
      
      Returns:
          [float] -- [the distance]
      """

      lat1 = 53.342998628
      lon1 = -6.256165642

      lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

      dlon = lon2 - lon1
      dlat = lat2 - lat1

      a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

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

      holiday_dates = ['9/8/19','10/8/19','11/8/19','12/8/19','13/8/19','14/8/19',
      '15/8/19','16/8/19','17/8/19','18/8/19','19/8/19','20/8/19','21/8/19','22/8/19',
      '23/8/19','24/8/19','25/8/19','26/8/19','27/8/19','28/8/19','29/8/19','30/8/19',
      '31/8/19','28/10/19','29/10/19','30/10/19','31/10/19','20/12/19','21/12/19',
      '22/12/19','23/12/19','24/12/19','25/12/19','26/12/19','27/12/19','28/12/19',
      '29/12/19','30/12/19','31/12/19','1/1/20','2/1/20','3/1/20','4/1/20',
      '5/1/20','17/2/20','18/2/20','19/2/20','20/2/20','21/2/20','3/4/20','4/4/20',
      '5/4/20','6/4/20','7/4/20','8/4/20','9/4/20','10/4/20','11/4/20','12/4/20',
      '13/4/20','14/4/20','15/4/20','16/4/20','17/4/20','18/4/20','19/4/20',
      '20/4/20','28/6/20','29/6/20','30/6/20','31/6/20','1/7/20','2/7/20','3/7/20',
      '4/7/20','5/7/20','6/7/20','7/7/20','8/7/20','9/7/20','10/7/20','11/7/20',
      '12/7/20','13/7/20','14/7/20','15/7/20','16/7/20','17/7/20','18/7/20','19/7/20',
      '20/7/20','21/7/20','22/7/20','23/7/20','24/7/20','25/7/20','26/7/20','27/7/20',
      '28/7/20','29/7/20','30/7/20','31/7/20','1/8/20','2/8/20','3/8/20','4/8/20',
      '5/8/20','6/8/20','7/8/20','8/8/20','9/8/20','10/8/20','11/8/20','12/8/20',
      '13/8/20','14/8/20','15/8/20','16/8/20','17/8/20','18/8/20','19/8/20','20/8/20',
      '21/8/20','22/8/20','23/8/20','24/8/20','25/8/20','26/8/20','27/8/20','28/8/20',
      '29/8/20','30/8/20''31/8/20']

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
            while temp not in stops_latlon.keys():
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

      dataframe = 'df_' + self.line

      line_encodings_path = os.path.join(directory, dataframe)

      month_source = line_encodings_path + '/MONTH_encode_map.json'
      stop_id_source = line_encodings_path + '/STOPPOINTID_encode_map.json'
      cluster_source = line_encodings_path + '/cluster_encode_map.json'

      features = [self.month, self.stop_id, self.cluster]
      dict_sources = [month_source, stop_id_source, cluster_source]

      for i in range(len(dict_sources)):
         json_obj = open(dict_sources[i])
         json_obj = json_obj.read()
         dictionary = json.loads(json_obj)
         features[i] = dictionary[features[i]]


   def create_vector(self):

      """
      Instance method that create a prediction model input
      vector from user parameters and parameters generated in
      this class.
      """

      self.vector = np.array([int(self.planned_arrival), 
                              float(self.rain), 
                              float(self.temp),
                              float(self.distance_centre),
                              int(self.t_minus_1118),
                              int(self.stop_id), 
                              int(self.day_of_week), 
                              int(self.month), 
                              int(self.holiday),
                              int(self.cluster)])


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

      result = model.predict(vector.reshape(1, -1))

      return int(result)



def main():

   """ 
   Executes the program.
   Formats input parameters, generates a preduction.
   Returns the prediction.
   """

   parameters = sys.argv[1:]

   stop_lat_lon_dict = '/Users/davidodwyer/Documents/studyCS/Semester_3/data/derived_data/db_stop_latlon.json'

   cluster_dict = '/Users/davidodwyer/Documents/studyCS/Semester_3/data/derived_data/stops_to_cluster_300m_radius.json'

   labels_dir = '/Users/davidodwyer/Documents/studyCS/Semester_3/models_label_encodings'

   models_dir = '/Users/davidodwyer/Documents/studyCS/Semester_3/models'


   formatter = FormatInput(parameters, stop_lat_lon_dict, cluster_dict)

   formatter.derive_features()
   formatter.label_encode(labels_dir)
   formatter.create_vector()


   predictor = Predict()

   result = predictor.predict(models_dir, formatter.line, formatter.vector)

   return result



if __name__ == '__main__':
   main()