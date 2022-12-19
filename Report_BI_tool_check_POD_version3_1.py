import ultils as u
import pandas as pd
from IPython.display import clear_output



### ___________________________________________________ Main ____________________________________________________________  
def read_pipeline(url_agg:str, str_time_from_:str, str_time_to_:str, split_from_:str, split_to_:str):
  print('hello Ninja !!!' + str(pd.Timestamp.now()))
  # reading and preprecessing
  print('Phase 1: Reading Data and preprocessing' + '-'*100)
  big_frame = u.read_folder_pod_resultQA_in_month(str_time_from_, str_time_to_)
  df = u.pre_processing(big_frame.loc[(pd.to_datetime(big_frame['attempt_date']) >= pd.Timestamp(str_time_from_)) & (pd.to_datetime(big_frame['attempt_date']) <= pd.Timestamp(str_time_to_))])
  # dispute
  clear_output()
  u.sales_channel(df)
  print('Date collected: ', df['attempt_date'].unique())
  print('Phase 2: Disputing, and Groupby Driver counting' + '-'*100)
  df = u.final_dispute(df)
  u.spliting_file(df, split_from=split_from_, split_to=split_to_)
  # pre-compute phase
  clear_output()
  print(df['attempt_date'].unique())
  print("Number of Unique Driver_name: ", df['driver_name'].nunique())
  print("Number of Unique Driver_type: ", df['driver_type'].value_counts())
  print("Number of Uni Hub name: ", df['hub_name'].nunique())
  print("Shape: ", df.shape)
  print(df.info())
  print('Phase 3: Mapping' + '-'*100)
  driver = u.mapping_phase(df, url_agg)
  reason =  u.export_final_reason_file(df.groupby(['first_attempt_date', 'reason']).apply(u.reason_fail_agg).reset_index())
  # compute phase
  clear_output()
  print('Final: ' + '-'*100)
  final = u.compute_phase(driver)
  u.export_final_driver_file(final)
  return df, final, reason