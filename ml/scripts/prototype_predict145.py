# this script is intended for as informal proof of concept
# for use by other team members for presentation
# coding standards are not adhered to 
# model deployment & prediction generation will not resemble this method

import os
import sys

import pandas as pd
from joblib import load

pd.options.mode.chained_assignment = None


# this dictionary is needed in the add_time_period()
# function and encode_dummy() function 

interval_map = {
0: '00:00 - 00:30',
1: '00:30 - 01:00',
2: '01:00 - 01:30',
3: '01:30 - 02:00',
4: '02:00 - 02:30',
5: '02:30 - 03:00',
6: '03:00 - 03:30',
7: '03:30 - 04:00',
8: '04:00 - 04:30',
9: '04:30 - 05:00',
10: '05:00 - 05:30',
11: '05:30 - 06:00',
12: '06:00 - 06:30',
13: '06:30 - 07:00',
14: '07:00 - 07:30',
15: '07:30 - 08:00',
16: '08:00 - 08:30',
17: '08:30 - 09:00',
18: '09:00 - 09:30',
19: '09:30 - 10:00',
20: '10:00 - 10:30',
21: '10:30 - 11:00',
22: '11:00 - 11:30',
23: '11:30 - 12:00',
24: '12:00 - 12:30',
25: '12:30 - 13:00',
26: '13:00 - 13:30',
27: '13:30 - 14:00',
28: '14:00 - 14:30',
29: '14:30 - 15:00',
30: '15:00 - 15:30',
31: '15:30 - 16:00',
32: '16:00 - 16:30',
33: '16:30 - 17:00',
34: '17:00 - 17:30',
35: '17:30 - 18:00',
36: '18:00 - 18:30',
37: '18:30 - 19:00',
38: '19:00 - 19:30',
39: '19:30 - 20:00',
40: '20:00 - 20:30',
41: '20:30 - 21:00',
42: '21:00 - 21:30',
43: '21:30 - 22:00',
44: '22:00 - 22:30',
45: '22:30 - 23:00',
46: '23:00 - 23:30',
47: '23:30 - 24:00',
48: '+24:00 - 00:30',
49: '+00:30 - 01:00',
50: '+01:00 - 01:30'
}



def load_model():
   
    """
    Function to load a model stored startically on app
    
    Returns:
    [trained linear regression model]
    """

    # find location of model

    file_path = '/Users/davidodwyer/Desktop' # to the directory
    file_name = 'original_mlr.joblib' 
    the_file = os.path.join(file_path, file_name)

    # load model

    model = load(the_file)

    return model


def read_input():

    """Function to read the input parameters
    
    Returns:
        [pandas dataframe]
    """

    parameters = sys.argv[1:]
    
    stop_id = parameters[0]
    month = parameters[1]
    day_of_week = parameters[2]
    rain = parameters[3]
    temp = parameters[4]
    planned_arrival = parameters[5]

    df = pd.DataFrame({
        'stop_id': [stop_id],
        'day': [day_of_week],
        'month': [month],
        'rain': [rain],
        'temp': [temp],
        'planned_arrival': [planned_arrival]
    })

    df = df.astype({
        'stop_id': 'int32',
        'day': 'int32',
        'month': 'int32',
        'rain': 'float64',
        'temp': 'float64',
        'planned_arrival': 'int32'
    })

    return df


def add_time_period(df):

    """Function to create time-period feature
    
    Returns:
        [dataframe with new feature]
    """

    # determine in which half hour period of the day the 
    # predicted time of arrival falls

    interval = df.iloc[0].planned_arrival // 1800 

    # find string representation of period from dict. mapping (top)

    inverval_string = interval_map[interval]

    # add the feature

    df['TIME_PERIOD_ARRIVAL'] = inverval_string

    # set the dtype

    df.TIME_PERIOD_ARRIVAL = df.TIME_PERIOD_ARRIVAL.astype('category') 

    return df


def is_holiday(df):

    """Function to add feature "school_off". Will be renamed in
    further dev., as it covers school, bank, and public holidays.
    Values will be representative of 2019.
    
    Returns:
        [dataframe with new feature]
    """

    # make list of public holidays

    holidays = [
    '1/1',
    '2/1',
    '3/1',
    '4/1',
    '5/1',
    '6/1',
    '7/1',
    '15/2',
    '16/2',
    '17/3',
    '23/3',
    '24/3',
    '25/3',
    '26/3',
    '27/3',
    '28/3',
    '29/3',
    '30/3',
    '31/3',
    '1/4',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '1/5',
    '7/5',
    '4/6',
    '6/8',
    '29/10',
    '30/10',
    '31/10',
    '1/11',
    '2/11',
    '21/12',
    '22/12',
    '23/12',
    '24/12',
    '25/12',
    '26/12',
    '27/12',
    '28/12',
    '29/12',
    '30/12',
    '31/12'
    ]

    # make date from day and month features    

    date = str(df.iloc[0].day) + '/' + str(df.iloc[0].month)

    # check if the date is in the list
    # 1 if yes, 0 if no

    if date in holidays:
        df['SCHOOL_OFF'] = 1
    else:
        df['SCHOOL_OFF'] = 0
        
    # set dtype

    df.SCHOOL_OFF = df.SCHOOL_OFF.astype('category')

    return df


def dict_key_from_item(dictionary, value):

    """Helper function for the encode_dummy function.
    Returns a key from a dictionary for a given value.
    Presupposes that there is only one key for a unique value.
    
    Returns:
        [int] -- [dictionary key]
    """

    # iterate dictionary entries
    # if value equals parameter value, return key

    for item in dictionary.items():
        if item[1] == value:
            return item[0]


def encode_dummy(df): 

    """A function that creates dummy variables for the
    categorical data types. Creates dummies with values 0
    for each value of the hard-coded categorical features.
    Finds the appropriate dummy bin given the input value, 
    changes the corresponding value to 1. 
    
    Returns:
        [Transformed dataframe, with dummied features]
    """

    # find current day, month, time of day period

    day = df.day[0]
    month = df.month[0]
    interval = df.TIME_PERIOD_ARRIVAL[0]

    # create month dummies (number months -1)

    for m in [2,3,4,5,6,7,8,9,10,11,12]:
        feat_name = 'month_' + str(m)
        df[feat_name] = 0

    # create day dummies (n-1)

    for d in [1,2,3,4,5,6]:
        feat_name = 'day_' + str(d)
        df[feat_name] = 0

    # create day period (i.e. which half hour segment) 
    # dummies (n-1)

    for p in [x for x in range(1,51)]:
        feat_name = 'period_' + str(p)
        df[feat_name] = 0

    # check the value doesn't correspond to the excluded dummy column
    # format the dummy column name, from prediction parameter
    # change the value of that cell

    if day != 0:
        target_day = 'day_' + str(day)
        df[target_day][0] = 1

    # repeat above procedure for month

    if month != 0:
        target_month = 'month_' + str(month)
        df[target_month][0] = 1

    # repeat as above
    # employ helper function to find key from interval value
    # use key value in formatting dummy column name 

    if interval != '00:30 - 01:00':
        key = dict_key_from_item(interval_map, interval)
        period = 'period_' + str(key)
        df[period][0] = 1
    
    # drop the features from which dummies were derived

    df.drop(['day', 'month', 'TIME_PERIOD_ARRIVAL'],\
         axis=1, inplace=True)

    return df


def order_columns(df):

    """Arranges the columns in the order the model expects
    """

    df = df[[
        'planned_arrival',
        'rain',
        'temp',
        'month_2',
        'month_3',
        'month_4',
        'month_5',
        'month_6',
        'month_7',
        'month_8',
        'month_9',
        'month_10',
        'month_11',
        'month_12',
        'day_1',
        'day_2',
        'day_3',
        'day_4',
        'day_5',
        'day_6',
        'period_1',
        'period_2',
        'period_3',
        'period_4',
        'period_5',
        'period_6',
        'period_7',
        'period_8',
        'period_9',
        'period_10',
        'period_11',
        'period_12',
        'period_13',
        'period_14',
        'period_15',
        'period_16',
        'period_17',
        'period_18',
        'period_19',
        'period_20',
        'period_21',
        'period_22',
        'period_23',
        'period_24',
        'period_25',
        'period_26',
        'period_27',
        'period_28',
        'period_29',
        'period_30',
        'period_31',
        'period_32',
        'period_33',
        'period_34',
        'period_35',
        'period_36',
        'period_37',
        'period_38',
        'period_39',
        'period_40',
        'period_41',
        'period_42',
        'period_43',
        'period_44',
        'period_45',
        'period_46',
        'period_47',
        'period_48',
        'period_49',
        'period_50',
        'SCHOOL_OFF',
        'stop_id'
    ]]

    return(df)


def scale_continous(df):

    """Uses the fitted sklearn standard scaler from model training
    to scale the continuous feature inputs.
    
    Returns:
        [standardly scaled continuous inputs]
    """

    # locate the scaler object, and load it

    file_path = '/Users/davidodwyer/Desktop' # to the directory
    file_name = 'basic_mlr_145_scaler.joblib' 
    the_file = os.path.join(file_path, file_name)

    scaler = load(the_file)

    # create a sub-dataframe of non-continuous features

    non_continuous_features = df[['stop_id', 'month', 'day', \
        'TIME_PERIOD_ARRIVAL', 'SCHOOL_OFF']] 

    # create list of continuous feautres

    continuous_features = ['planned_arrival', 'rain', 'temp']

    # scale the continuous features, form as new dataframe

    scaled_continuous = pd.DataFrame(scaler.transform(df[continuous_features])\
        , columns=continuous_features)

    # reset the sub-dataframes indices

    scaled_continuous.reset_index(drop=True, inplace=True)

    non_continuous_features.reset_index(drop=True, inplace=True)

    # join the sub-dataframes / rejoin to reform the "original"

    join = pd.concat([scaled_continuous, non_continuous_features]\
        , axis=1)

    # reset the index

    join.reset_index(drop=True, inplace=True)

    return join


def run():

    """
    Function to run script

    """

    df = read_input() # the parameters
    df = add_time_period(df) # a feature
    df = is_holiday(df) # a feature
    df = scale_continous(df) # continous feature transformation
    df = encode_dummy(df) # categorical feature transformation
    df = order_columns(df) # ordering model inputs
    model = load_model() # the multiple linear regression model
    prediction = int(model.predict(df)) # form a prediction
    return prediction # return the prediction


if __name__ == "__main__":

    run()



