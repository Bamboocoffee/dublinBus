# worker script. 
# compiled from notebook workflow & testing

import glob
import os

import pandas as pd

# read in the leavetimes dataframe

try:

   dataframe_path = '/tmp/ssh_mount/dataframes'

   os.chdir(dataframe_path)

   df = pd.read_feather('020719_postassessquality_leavetimes.feather')
   print('Leavetimes loaded')

except:
   print('Error Loading Leavetimes')


# change trip_id to integer for comparison with indices

try:

   df.TRIPID = df.TRIPID.astype('int32')
   print('Successful TRIPID dtype conversion')

except:

   print("Error Changing Leavetimes Dtype")


# get paths and line_ids from trip indices csv files

try:
   indices_path = '/tmp/ssh_mount/trip_indices'

   os.chdir(indices_path)

   files = glob.glob('*.csv')
   files_paths = [os.path.join(os.getcwd(), f) for f in files]
   line_names = [file.strip('.csv') for file in files]

   print('Trip Indices Collected')

except:
   print("Error collecting trip indices")


try:
   destination_path = '/tmp/ssh_mount/lines'
   os.mkdir(destination_path)
   print('Destination directory created.')
except:
   print("Directory not created")

os.chdir(destination_path)

count=0
failed_lines = []

for file in files_paths:

   try:

      print("Starting creation of line " + line_names[count] + 'sub-dataframe')

      trip_indices = pd.read_csv(file, header=None, squeeze=True, index_col=0)

      if trip_indices.dtype != 'int32':
         trip_indices = trip_indices.astype('int32')

      print("Creating chunks for line " + line_names[count]) 

      chunks = []

      for i, v in trip_indices.items():
         chunks.append(df.loc[df.TRIPID == v, :])

      print("Concatenating chunks for " + line_names[count]) 

      sub_df = pd.concat(chunks)

      del chunks

      sub_df.reset_index(drop=True, inplace=True)

      filename = 'df_' + line_names[count] + '.feather'

      sub_df.to_feather(filename)

      del sub_df

      count+=1

      print("Sub-dataframe for line" + line_names[count] + 'finished')

   except:
      print("Failed at line " + line_names[count])
      failed_lines.append(line_names[count])
