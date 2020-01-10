# worker script. 
# compiled from notebook workflow & testing

import os

import pandas as pd

# make a directory to save the sub-dataframes to

# try:
destination_path = '/tmp/ssh_mount/trip_indices'
#    os.mkdir(destination_path)
# except:
#    print("Directory not created")


# move directory to where the parent dataframe is

try:
   source_path = '/tmp/ssh_mount/dataframes'
   os.chdir(source_path)

except:
   print("Error Changing Paths")


# load the dataframe

try:
      df = pd.read_feather('230719_trips.feather')

      line_ids = df.LINEID.unique().tolist()
      print("Line List Created...")

except:

      print("Error Loading Dataframe")


# create the sub-dataframes by LINEID



# initialise a counter

count = 1
errors = 0
error_lines = []

# iterate through the line_id list

for line in line_ids:

      try:

            print("Creating sub-df #", count)

            # select rows where the line_id is specified

            df_sub = df[df.LINEID == line]

            print("Sub-df", count, "created. ")

            # reset the index on new sub-dataframe

            df_sub.reset_index(drop = True, inplace=True)

            # drop nulls

            df_sub.dropna(axis=0, inplace=True)

            # drop rows where departure is later than arrival

            df_sub.drop(df_sub[df_sub.ACTUALTIME_DEP \
                  > df_sub.ACTUALTIME_ARR].index, inplace=True)

            # drop rows where the journey time is under 45min

            df_sub.drop(df_sub[df_sub.ACTUAL_TRIP_DURATION\
                  < 2750].index, inplace=True)

            # determine 99.87 percentile / 3 std from mean value

            ceiling = int(df_sub.ACTUAL_TRIP_DURATION.quantile(.9987))

            # drop outlier rows (above ceiling)

            df_sub.drop(df_sub[df_sub.ACTUAL_TRIP_DURATION\
                  > ceiling].index, inplace=True)
            
            # make list (Series) of unique ids

            df_sub_trip_ids = df_sub.TRIPID.drop_duplicates()

            # change from categorical to int

            df_sub_trip_ids.astype('int32')

            # save indices

            filename = line + '.csv'

            save_path = os.path.join(destination_path, filename)

            df_sub_trip_ids.to_csv(save_path)

            del df_sub

            print("Trip indices", count, "created & saved")

      except:
            print("Error occurred at on the", count, "iteration")
            errors +=1
            error_lines.append(line)

      count+=1

print('\n\n************** Finished')
print(errors, "errors")
print('Errors on lines:')
for error in error_lines:
      print(error)
      


