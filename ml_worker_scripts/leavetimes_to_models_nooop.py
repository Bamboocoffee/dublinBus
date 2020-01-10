import os
import sys
import glob
import json
import pandas as pd
import numpy as np
from math import sin, cos
from pympler import asizeof
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
# from haversine import haversine, Unit
from sklearn.externals import joblib
from joblib import dump, load



def change_directory(directory_path):
      try:
            os.chdir(directory_path)
            print("Directory Changed")
      except:
            print("Failed: Change Directory")

def get_dataframes():

      try:
            files = glob.glob('*.feather')

            files_paths = [os.path.join(os.getcwd(), f) for f in files]

            line_names = [f.strip('.feather') for f in files]

            print("Returned dataframe paths, dataframe lines")

            return files_paths, line_names

      except:
            print("Failed: Getting Dataframes")

def make_abt(crude_df, ln):

      line_name = ln
      df = crude_df

      try:

            df = add_weather(df)

            df = add_stop_lat_lon_features(df)

            df = add_distance_from_centre_feature(df)

            df = add_day_of_week_acyclic(df)

            df = add_month_acyclic(df)

            df = add_days_since_year_start(df)

            df = is_a_holiday(df)

            df = add_stop_cluster(df)

            df = drop_cols(df, [
                  'DAYOFSERVICE',
                  'TRIPID',
                  'PROGRNUMBER',
                  'PLANNEDTIME_DEP',
                  'ACTUALTIME_DEP',
                  'VEHICLEID',
                  'stop_lat',
                  'stop_lon'
                  ])

            df = set_dtypes(df, {
                  'STOPPOINTID':'category',
                  'PLANNEDTIME_ARR':'int32',
                  'rain':'float32',
                  'temp':'float32',
                  'distance_centre':'float32',
                  'DAYOFWEEK':'category',
                  'MONTH':'category',
                  'day_into_year':'int32',
                  'holiday':'category',
                  'cluster':'category'
                  })

            df = add_label_encoding(df, [
                  'STOPPOINTID',
                  'DAYOFWEEK',
                  'MONTH',
                  'holiday',
                  'cluster'
                  ], line_name=line_name)

            return df

      except:

            print("Problem with make ABT function")

def make_model(X,y):

      print(X.shape)
      print(y.shape)

      try:
      
            model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0, warm_start=False)

            print('Model initialised')

            model.fit(X, y.values.ravel())

            print("model fit")

            return model

      except:

            print("Problem with make_model function")

def add_weather(df):

      try:
    
            weather = weather_base.copy()

            # create time and hour columns for ease of joining with df

            dates, hours = zip(*[(d.date(), d.time()) for d in weather['date']])
            weather = weather.assign(dates=dates, hours=hours)

            # change type

            weather.hours = weather.hours.astype('str')

            # define mapping for weather hour feature

            replacements_hours = {
            '00:00:00' : '0',
            '01:00:00' : '1',
            '02:00:00' : '2',
            '03:00:00' : '3',
            '04:00:00' : '4',
            '05:00:00' : '5',
            '06:00:00' : '6',
            '07:00:00' : '7',
            '08:00:00' : '8',
            '09:00:00' : '9',
            '10:00:00' : '10',
            '11:00:00' : '11',
            '12:00:00' : '12',
            '13:00:00' : '13',
            '14:00:00' : '14',
            '15:00:00' : '15',
            '16:00:00' : '16',
            '17:00:00' : '17',
            '18:00:00' : '18',
            '19:00:00' : '19',
            '20:00:00' : '20',
            '21:00:00' : '21',
            '22:00:00' : '22',
            '23:00:00' : '23'
            }
      
            # perform mapping

            weather.hours = weather.hours.map(replacements_hours)

            # change weather hours to int, dates to datetime

            weather.hours = weather.hours.astype('int')

            weather.dates = weather.dates.astype('datetime64[ns]')

            # give df an hour column for joining

            df['hour'] = df.apply(lambda row: int(row['PLANNEDTIME_ARR'] / 3600), axis=1)
            df.loc[df.hour == 24, 'hour'] = 0
            df.loc[df.hour == 25, 'hour'] = 0

            # join the dataframes

            df_merged = pd.merge(df, weather, how='left', \
                        left_on=['DAYOFSERVICE', 'hour'], right_on=['dates', 'hours'])

            # drop nans (one single row for weather 31/Dec in weather)

            df_merged.dropna(how='any', axis=0, inplace=True)

            # tidy: remove joining cols

            df_merged.drop([
                  'hours',
                  'hour',
                  'dates',
                  'date',
            ], axis=1, inplace=True)

            print('— weather added')

            return df_merged

      except:

            print("Problem with add_weather function")

def add_stop_lat_lon_features(df):

      try:
    
            df['stop_lat'] = df.STOPPOINTID
            df['stop_lon'] = df.STOPPOINTID

            df = df.astype({
                  'stop_lat':'int32',
                  'stop_lon':'int32'
            })
            

            # for stops in dictionary and df, map; retain values
            # for stops not in dictionary
            
            df['stop_lat'] = df['stop_lat'].map(stops_lat).fillna(df['stop_lat'])
            df['stop_lon'] = df['stop_lon'].map(stops_lon).fillna(df['stop_lat'])
                  
            # stops in the dublin but dataset and in the mapping dict
            stops_in_current_dataframe = list(set(df.STOPPOINTID.unique()).intersection(set(stops_latlon.keys())))
            
            # stops in dublin bus dataset but not in mapping dict
            stops_not_in_dict = list(set(df.STOPPOINTID.unique()).difference(set(stops_latlon.keys())))
            
            for stop in stops_not_in_dict:
                  while stop not in stops_in_current_dataframe:
                        if stop < 7692:
                              stop += 1
                        else:
                              while stop not in stops_in_current_dataframe and stop > 2:
                                    stop -= 1
                  df.loc[df.STOPPOINTID == stop, 'stop_lat'] = stops_latlon[stop][0]
                  df.loc[df.STOPPOINTID == stop, 'stop_lon'] = stops_latlon[stop][1]

            print("— latitudes and longitudes added")
            
            return df

      except:

            print("Problem with add_stop_lat_lon function")

# def distance_from_city_centre(row):
    
      # """
      # Calculates the distance of a lat/lon pair from the city centre
      # defined as O' Connell bridge, Dublin.

      # Wrapped in def add_distance_from_centre_feature()
      # """
      
      # try:

      #       centre_gps = (53.342998628, -6.256165642)

      #       stop_gps = (row.stop_lat, row.stop_lon)

      #       distance = haversine(centre_gps, stop_gps)

      #       return distance

      # except:

      #       print("Problem with distance from city centre helper func.")
    
def haversine(row):

      """
      Taken from: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
      """

      lat1 = 53.342998628
      lon1 = -6.256165642

      lat2 = row.stop_lat
      lon2 = row.stop_lon

      lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

      dlon = lon2 - lon1
      dlat = lat2 - lat1

      a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

      c = 2 * np.arcsin(np.sqrt(a))
      km = 6367 * c
      
      return km

def add_distance_from_centre_feature(df):
    
      """
      Creates a new feature: distance of each row's stop from
      the city centre
      """

      try:

            df['distance_centre'] = df.apply(haversine, axis=1)

            print('— distance from centre added')

            return df

      except:

            print("Problem with distance_from_centre function")

def add_day_of_week_acyclic(df):

      try:
    
            df['DAYOFWEEK'] = df.DAYOFSERVICE.dt.dayofweek

            print('— day of week added')

            return df

      except:

            print("Problem with add_day_of_week_function")

def add_month_acyclic(df):

      try:

            df['MONTH'] = df.DAYOFSERVICE.dt.month

            print('— month added')

            return df

      except:

            print("Problem with add_month_acyclic function")

def add_days_since_year_start(df):
    
      """
      Adds feature. Integer representing day of year to each input.
      Encodes the desired cumulative proximity-to-start-of-year.
      """

      try:
    
            df['day_into_year'] = df.DAYOFSERVICE.dt.dayofyear

            print('— days since 1/1/18 added')
            
            return df

      except:

            print("Problem with add_day_since_year_start function")

def is_a_holiday(df):
    
      """
      creates a feature indicating (with '1') a) primary 
      school not in session b) bank holiday c) public
      holiday.
      """

      try:

            # school, public, bank holidays

            holiday_dates = [
            '1/1/18', '2/1/18', '3/1/18', '4/1/18', '5/1/18', '6/1/18',
            '7/1/18', '15/2/18', '16/2/18', '17/3/18', '23/3/18',
            '24/3/18', '25/3/18', '26/3/18', '27/3/18', '28/3/18',
            '29/3/18', '30/3/18', '31/3/18', '1/4/18', '2/4/18', '3/4/18',
            '4/4/18', '5/4/18', '6/4/18', '7/4/18', '8/4/18', '1/5/18',
            '7/5/18', '4/6/18', '6/8/18', '29/10/18', '30/10/18',
            '31/10/18', '1/11/18', '2/11/18', '21/12/18', '22/12/18',
            '23/12/18', '24/12/18', '25/12/18', '26/12/18', '27/12/18',
            '28/12/18', '29/12/18', '30/12/18', '31/12/18']

            # make list of holidays into datetime series

            holiday_dates = pd.to_datetime(holiday_dates)

            # define and create a range of datetimes
            # covering the summer break

            start = pd.to_datetime('28/6/2018')
            end = pd.to_datetime('1/Sep/2018')
            summer_break = pd.DatetimeIndex(start=start, end=end, freq='d')

            # join the holiday date sets

            holiday = summer_break.append(holiday_dates)

            # initialise a new feature: 0, not a holiday

            df['holiday'] = '0'

            # find all row numbers corresponding to a holiday

            holiday_rows = []

            for item in holiday:
                  temp_list = df.loc[df.DAYOFSERVICE == item].index.to_list()
                  for i in temp_list:
                        holiday_rows.append(i)
      
            # change the value of rows with holiday date        
                        
            df.loc[holiday_rows, "holiday"] = '1'

            print('— holiday marker added')
      
            return df

      except:

            print("Problem with add_holiday function")

def add_stop_cluster(df):

      try:

            json_obj = open(stop_cluster_json_path)
            json_obj = json_obj.read()
            cluster_map_original = json.loads(json_obj)

            cluster_map = {}

            for k, v in cluster_map_original.items():
                  for item in v:
                        cluster_map[item] = k
                        
            df['cluster'] = df.STOPPOINTID.map(cluster_map)

            # fix for belatedly noticed data quality issue
            # affecting derived data: stop_lat_lon

            # for any stops not in the dictionary
            # roll back to closest stop both in dictionary
            # and dataset. impute the value

            missing_stops = df.loc[df.cluster.isna(), 'STOPPOINTID'].unique().tolist()

            for stop in missing_stops:
                  temp = stop
                  available_stops = set(df.STOPPOINTID.unique().tolist()).intersection(stops_latlon.keys())
                  while temp not in available_stops:
                        if temp < 7692:
                              temp += 1
                        else:
                             while temp not in available_stops and temp > 2:
                                   temp -= 1 
                  impute_val = df.loc[df.STOPPOINTID == temp, 'cluster'].unique()[0]
                  df.loc[df.STOPPOINTID == stop, 'cluster'] = impute_val
                  # print('_'*50)
                  # print('\n')

            print('— stop clusters added')

            return df

      except:

            print("Problem with add_stop_cluster function")
        
def add_label_encoding(df, categoricals_to_encode, line_name):
    
      """
      Performs label-encoding of categorical features.
      Suitable for tree-based methods that do not 
      add ordinality meaning.

      INPUT: dataframe and list of features to encode

      OUTPUT: dataframe with new encoded features; 
            originals removed.
            
      USE: run on total data after all experiments 
      for map of all values > encodings.

      """
      label_encode_dir = '/Users/davidodwyer/Documents/studyCS/Semester_3/models_label_encodings'
      line_name = line_name

      # iterate through features list
      # label-encode their values

      try:
    
            for feature in categoricals_to_encode:
                  le = LabelEncoder()
                  name = feature + '_label'
                  df[name] = le.fit_transform(df[feature])
            
                  # create a mapping of original to labelled
                  # values

                  encoding = df[name].tolist()
                  original = le.inverse_transform(df[name]).tolist()
            
                  dictionary = dict(zip(original, encoding))


                  # save the mapping to disk, as json

                  line_encode_dir = label_encode_dir + '/' + line_name
            
                  # line_encode_dir = os.path.join(label_encode_dir, line_name)

                  if not os.path.isdir(line_encode_dir):
                        os.mkdir(line_encode_dir)


                  dict_name = feature + '_encode_map.json'
                  save_target = os.path.join(line_encode_dir, dict_name)
            
                  with open(save_target, 'w') as json_file:
                        json.dump(dictionary, json_file)
            


            # drop the original, unencoded features        
                        
            for feature in categoricals_to_encode:
                  df.drop(feature, axis=1, inplace=True)

            print('— features label encoded')
                  
            return df

      except:

            print("Problem with label encoding function")

def drop_cols(df, cols):

      try:
    
            df.drop(cols, axis=1, inplace=True)

            print('— unneeded columns dropped')
            
            return df

      except:

            print("Problem with drop_cols function")

def set_dtypes(df, type_dictionary):

      try:
    
            df = df.astype(type_dictionary)

            print('— dtypes set')

            return df

      except:

            print("problem with set_dtypes function")

def main():

      try:
            change_directory(dataframe_dir)

            dataframe_paths, line_numbers = get_dataframes() 

            number_lines = len(line_numbers)

            print("Directory Found. Dataframes Indexed.")

      except:
            print("Failed: Getting Dataframes")

      i=0

      while i < number_lines:

            problematic_iterations = []

            try:

                  line_name = str(line_numbers[i])
                  dataframe_path = dataframe_paths[i]

                  print("Preparing line", line_name, "for modelling...")

                  df = pd.read_feather(dataframe_path)

                  ts = pd.Timestamp('31/Dec/2018')

                  df.drop(df.loc[df.DAYOFSERVICE == ts].index, axis=0, inplace=True)

                  print("...Dataframe loaded")
                  print("...Making ABT")

                  X = df.loc[:, df.columns != 'ACTUALTIME_ARR']

                  y = df.loc[:, df.columns == 'ACTUALTIME_ARR']

                  # y = y.iloc[:,0]

                  X = make_abt(X, line_name)

                  print("...ABT created")

                  print("...Fitting model for line", line_name)

                  print(X.head())

                  model = make_model(X,y)

                  model_name = line_name + '.joblib'

                  save_model_path = os.path.join(model_save_dir, model_name)

                  if asizeof.asizeof(model) > 50:

                        joblib.dump(model, save_model_path, compress=True)

                  else:

                        problematic_iterations.append(i)

                  print("Loop finished...")
                  print(i+1, 'of', number_lines, 'models attempted')
                  print(len(problematic_iterations), "problems thus far.")
                  print('_'*50)
                  print('\n\n')

                  i+=1

            except:

                  print("Modelling Failed on", i, "of", number_lines)
                  problematic_iterations.append(i)
                  print("Continuing with", i+1)
                  i+=1

      print("Modelling Completed.")
      if len(problematic_iterations) > 0:
            print(len(problematic_iterations), "unmade models on iterations:")
            for item in problematic_iterations:
                  print(item, sep=', ', end='')



if __name__ == '__main__':

      try:

            dataframe_dir = '/tmp/ssh_mount/dataframes/lines'
            model_save_dir = '/Users/davidodwyer/Documents/studyCS/Semester_3/models'
            label_encode_dir = '/Users/davidodwyer/Documents/studyCS/Semester_3/models_label_encodings'
            weather_dataframe_path = '/Users/davidodwyer/Documents/studyCS/Semester_3/data/dataframes/L145/weather.feather'
            stops_latlon_path = '/Users/davidodwyer/Documents/studyCS/Semester_3/data/derived_data/db_stop_latlon.json'
            stop_cluster_json_path = '/Users/davidodwyer/Documents/studyCS/Semester_3/data/derived_data/stops_to_cluster_300m_radius.json'

            # get weather dataframe
            weather_base = pd.read_feather(weather_dataframe_path)

            # get busstop latitude-longitude dictionary
            json_obj = open(stops_latlon_path)
            json_obj = json_obj.read()
            stops_latlon = json.loads(json_obj)
            stops_latlon = {int(k): stops_latlon[k] for k in stops_latlon.keys()}
            stops_lat = {k:v[0] for k, v in stops_latlon.items()}
            stops_lon = {k:v[1] for k, v in stops_latlon.items()}
            print('\n\n\n')
            print("Set-up Suceeded...")

      except:

            print("Failed: Setup. Weather and Lat/Lon Data not Loaded")  

      main()



