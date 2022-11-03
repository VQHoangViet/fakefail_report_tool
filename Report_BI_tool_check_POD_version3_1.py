print('hello Ninja3_1!!!')

import os
import glob
from google.colab import auth
import gspread
from google.auth import default
import datetime as dt
from google.colab import drive
import pandas as pd
import numpy as np
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from IPython.display import clear_output
from os import listdir
from os.path import isfile, join
import os
from dateutil.relativedelta import relativedelta

drive.mount('/content/drive', force_remount=True)
pd.set_option('display.max_columns', None)
auth.authenticate_user()

# module:
def get_first_attempt_date(x):
  try:
    x = x.drop(columns='first_attempt_date')
  except:
    pass
  first_attempt_date = x.groupby('tracking_id')['attempt_date'].min().reset_index().rename(columns={'attempt_date': 'first_attempt_date'})
  x = x.merge(first_attempt_date, how='inner', left_on='tracking_id', right_on='tracking_id')
  return x



# Phase1: reading data
def reading_last_7_day():
  url = [
    'https://docs.google.com/spreadsheets/d/11R3RKcyleeHDQPr2y3QpiOLReyc5_T8h2DNwS2fEYlA/edit#gid=0', ## thu 2
    'https://docs.google.com/spreadsheets/d/1uKSr4flvsSvejRHPid1Ja718Ozl9XFXZAuxi2mVYbKI/edit#gid=0', ## thu3
    'https://docs.google.com/spreadsheets/d/1XEXt04xtTfsdX3D0rLRriO2XpOxDLI93j9DX2YW1Fys/edit#gid=0', ## thu 4
    'https://docs.google.com/spreadsheets/d/1U9PeYtVqxQOjl5za9UfxuSZmcKC3U_Z3Y9Ym1kxMhyM/edit#gid=0', ## Thu 5
    'https://docs.google.com/spreadsheets/d/1Q-igQBq0CnR14PWokk4GZcOn3iJ7RboUKFSp6bMiOcg/edit#gid=0', ## Thu 6
    'https://docs.google.com/spreadsheets/d/1kPl5-R6FpuRLiNvVmicSQCMYorXqWkUx1V9Frdj8_Ig/edit#gid=0',  ## thu 7
    'https://docs.google.com/spreadsheets/d/1Z7wGCT4Sh8mQl69nSMZxDuMbbaWBEbWewlWQr_xm9T0/edit', ## cn 

  ]

  bi_data_raw = pd.DataFrame()
  for i in url:
    # holding temp data
    print(i)
    creds, _ = default()
    gc = gspread.authorize(creds)
    temp = gc.open_by_url(i).worksheet("Sheet1")
    # Convert to a DataFrame and render.
    bi_data_raw = pd.concat([bi_data_raw, get_as_dataframe(temp)])
  return bi_data_raw.dropna(how='all', axis=1).dropna(how='all', axis=0)

def read_folder_pod_resultQA_in_month():
  # Get data file names
  mypath = '/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'
  source_df = pd.DataFrame({
    'filename': [mypath + f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))],
    'date' : pd.to_datetime(pd.Series([item.replace(".csv", "") for item in [f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))]]))
  })
  print(source_df['date'])
  needed_df = source_df.loc[source_df.date >= pd.Timestamp(dt.datetime.now().year, (dt.datetime.now().date() - relativedelta(months=2)).month, 26)] # select continually update date range

  # get data frame
  dfs = []
  print(needed_df['date'].unique())
  for filename in needed_df['filename']:
      x = pd.read_csv(filename)
      print('Path File:{}, duplicated :{}'.format(filename, x[x['waypoint_id'].duplicated()].shape))
      dfs.append(x)
  big_frame = pd.concat(dfs, ignore_index=False)  # Concatenate all data into one DataFram
  print(big_frame.shape)
  return big_frame

# Phase 2: pre-processing, dispute
def driver_finder(x):
    if 'NEXT' in x: return 'fulltime'
    elif 'FRLA' in x: return 'freelancer'
    elif( 'FTS' in x) | ('AGAR' in x) | ('189-FRLA' in x) | ('518-FRLA' in x) | ('XDOCT' in x) | ('TSS' in x) | ('GRAB' in x) | ('RAGA' in x) |('AHA' in x) : return '3PLs'
    else: return 'others'

def pre_processing(x):
    # save version data:
    try:
      x.drop(columns=['Unnamed: 0'], inplace=True)
    except:
      pass
    x.attempt_datetime = pd.to_datetime(x.attempt_datetime)
    x = get_first_attempt_date(x) ## 11/03/2022: get first attempt to mapping
    # notice: no_call_log_aloninja = fakefail (update: 30/09/2022)
    x['no_call_log_aloninja'] = 0
    x.loc[x['count_call_log'].isna(), 'no_call_log_aloninja'] = 1


    # CONVERT data_type
    x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
    x[['hub_id', 'order_id', 'waypoint_id']] = x[['hub_id', 'order_id', 'waypoint_id']].astype('int64')
    x[['reason_no' , 'Thời gian ghi nhận fail attempt phải trước 10PM' ,
        'Cuộc gọi phải phát sinh trước 8PM',
        'Lịch sử tối thiểu 3 cuộc gọi',
        'Thời gian giữa mỗi cuộc gọi tối thiểu 1 phút',
        'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại',
        'Thời gian đổ chuông >10s trong trường hợp khách không nghe máy' ,
        'No Record', 'no_call_log_aloninja']] = x[['reason_no' , 'Thời gian ghi nhận fail attempt phải trước 10PM' ,
        'Cuộc gọi phải phát sinh trước 8PM',
        'Lịch sử tối thiểu 3 cuộc gọi',
        'Thời gian giữa mỗi cuộc gọi tối thiểu 1 phút',
        'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại',
        'Thời gian đổ chuông >10s trong trường hợp khách không nghe máy' ,
        'No Record', 'no_call_log_aloninja']].replace('-', 0).astype('float64')
    # dropna columns  
    x = x.dropna(how='all', axis=1).dropna(how='all', axis=0)   

    # fill na to zero
    x['count_call_log'] = x['count_call_log'].fillna(0)

    # drop dupli waypoint_id AND tracking_id
    # bug: drop được duplicate nhưng bị mất FF attempt nếu nằm ở đầu (fixed)
    x = x.sort_values(['attempt_datetime'])
    x = x.drop_duplicates(subset=['order_id', 'waypoint_id'], keep='last')

    # find driver type
    x['driver_type'] =  x.driver_name.apply(driver_finder)
    return pd.DataFrame(x)

def get_disputetime():
    return (dt.datetime.now() - dt.timedelta(days=5)).date()

def final_dispute(x):

  ## Disputing
  x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
  x['final_result'] = 0
  url = [
    'https://docs.google.com/spreadsheets/d/1i1Rha9Qg1qZ9sGI0-ddX9QBlO6Jg9URy2tm62Fu3X20/edit#gid=1966091300',
    'https://docs.google.com/spreadsheets/d/1P0ohdLCGGvk037IHEFeiGvvc7l2bku5HIYCSgLT4i4o/edit#gid=419800374'
  ]

  disputing = pd.DataFrame()
  for i in url:
    # holding temp data
    print(i)
    creds, _ = default()
    gc = gspread.authorize(creds)
    temp = gc.open_by_url(i).worksheet("Detail")
    # Convert to a DataFrame and render.
    disputing = pd.concat([disputing, get_as_dataframe(temp, evaluate_formulas=True)[['order_id','waypoint_id', 'Status']]])


  disputing =  disputing.dropna(how='all', axis=1).dropna(how='all', axis=0).drop_duplicates(subset=['waypoint_id', 'order_id'])
  accepted_waypoint = disputing[disputing['Status'].isin(['Corrected', 'Product xin loại trừ', 'no'])]
  
  x['disputing'] = 0
  x['corrected_dispute'] = 0
  x.loc[x['waypoint_id'].isin(disputing['waypoint_id']), 'disputing'] = 1
  x.loc[x['waypoint_id'].isin(accepted_waypoint['waypoint_id']), 'corrected_dispute'] = 1

  ### Mass bug
  # Lỗi sever 
  x['affected_by_mass_bug'] = 0
  x.loc[(x.attempt_datetime >= pd.Timestamp(2022, 9, 28, 12, 00)) & (x.attempt_datetime <= pd.Timestamp(2022, 9, 29, 23, 59)),'affected_by_mass_bug'] = 1
  x.loc[(x.attempt_datetime >= pd.Timestamp(2022, 10, 5, 16, 30)) & (x.attempt_datetime <= pd.Timestamp(2022, 10, 5, 23, 59)),'affected_by_mass_bug'] = 1
  
  # Lỗi sever: Các đầu số Mobi của khách hàng không thể gọi đc callee == *đầu số mobi* từ 6h00 24/10/2022 -> 23h59 cùng ngày
  mobi_ = ['90' , '93' , '89' ,  '70',	'79',	'77',	'76',	'78']
  x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
  x['callee_'] = x['callee'].str[:2]
  x['driver_contact_'] = x['driver_contact'].str[2:4]

  x.loc[(  ( (x.attempt_datetime >= pd.Timestamp(2022, 10, 24, 6)) & (x.attempt_datetime <= pd.Timestamp(2022, 10, 24, 23, 59)) ) &
           (x['callee_'].isin(mobi_) | x['driver_contact_'].isin(mobi_))
        ), 'affected_by_mass_bug'] = 1

  # final:
  print(x[ (x['result'] == 'fake_fail') & ((x['affected_by_mass_bug'] == 0) & (x['corrected_dispute'] == 0)) ].shape)
  x.loc[ (x['result'] == 'fake_fail') & ((x['affected_by_mass_bug'] == 0) & (x['corrected_dispute'] == 0))   , 'final_result'] = 1

  ## Nocode here


  print('Done Dispute!')


  url = [
    'https://docs.google.com/spreadsheets/d/1KX1TceinNWG4_CCuN89VqXsLYnOquN-FnLIg_i1N-A8/edit#gid=0',
  ]

  disputing_after_final = pd.DataFrame()
  for i in url:
    # holding temp data
    print(i)
    creds, _ = default()
    gc = gspread.authorize(creds)
    temp = gc.open_by_url(i).worksheet("Detail")
    # Convert to a DataFrame and render.
    disputing_after_final = pd.concat([disputing_after_final, get_as_dataframe(temp, evaluate_formulas=True)[['tracking_id','waypoint_id', 'Status']]])


  disputing_after_final =  disputing_after_final.dropna(how='all', axis=1).dropna(how='all', axis=0).drop_duplicates(subset=['waypoint_id', 'tracking_id'])
  corrected_dispute_after_final_waypoint = disputing_after_final[disputing_after_final['Status'].isin(['no'])]
  
  x['corrected_dispute_after_final'] = 0
  x.loc[x['waypoint_id'].isin(corrected_dispute_after_final_waypoint['waypoint_id']), 'corrected_dispute_after_final'] = 1
  x.loc[(x.attempt_datetime >= pd.Timestamp(2022, 10, 1)) & (x.attempt_datetime <= pd.Timestamp(2022, 10, 15))].to_csv('/content/raw1_15_thang10.csv', index=False)
  x.loc[(x.attempt_datetime >= pd.Timestamp(2022, 10, 16)) & (x.attempt_datetime <= pd.Timestamp(2022, 10, 25))].to_csv('/content/raw16_25_thang10.csv', index=False)
  print('Done Dispute combine!')

  return pd.DataFrame(x)

def spliting_file(x):
  for i in x['attempt_date'].unique():
    x[x['attempt_date'] == i].to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'+str(i)+'.csv', index=False)
    print("Done file: " + str(i))
  print('DONE SLITING')

# Phase 3: grouping
def bi_agg(x):
    names = {
        # note: count by attempt
        'Thời gian ghi nhận fail attempt phải trước 10PM': x[x['Thời gian ghi nhận fail attempt phải trước 10PM']==1]['waypoint_id'].count(),
        'Cuộc gọi phải phát sinh trước 8PM': x[x['Cuộc gọi phải phát sinh trước 8PM']==1]['waypoint_id'].count(),
        'Lịch sử tối thiểu 3 cuộc gọi': x[x['Lịch sử tối thiểu 3 cuộc gọi']==1]['waypoint_id'].count(),
        'Thời gian giữa mỗi cuộc gọi tối thiểu 1 phút': x[x['Thời gian giữa mỗi cuộc gọi tối thiểu 1 phút']==1]['waypoint_id'].count(),
        'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại': x[x['Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại']==1]['waypoint_id'].count(),
        'Thời gian đổ chuông >10s trong trường hợp khách không nghe máy': x[x['Thời gian đổ chuông >10s trong trường hợp khách không nghe máy']==1]['waypoint_id'].count(),
        'no_call_log_aloninja': x[x['no_call_log_aloninja']==1]['waypoint_id'].count(),
        # 'no_call_log_aloninja': x[x['fail_pod_reason']=='no_call_log_aloninja']['waypoint_id'].count(),
        'No Record': x[x['No Record']==1]['waypoint_id'].count(),
        'BI_tracking_id': set(list(x[(x['result']=='fake_fail') & (x['affected_by_mass_bug']!=1) & (x['corrected_dispute']!=1)]['tracking_id'].unique())),
        'total_attempt': x['waypoint_id'].count(),
        'total_orders': x['order_id'].nunique(),
        # 'BI_FakeFail': x[x['result']=='fake_fail']['tracking_id'].nunique(), # OPEX dont compute by Waypoint 19/10/2022
        'MASS_BUG_PRODUCTION': x[(x['affected_by_mass_bug']==1) & (x['result']=='fake_fail')]['order_id'].nunique(),
        'disputing_orders':  x[(x['disputing']==1) &  (x['result']=='fake_fail')]['order_id'].nunique(),
        'correted_by_disputing_orders':  x[(x['corrected_dispute']==1) & (x['result']=='fake_fail')]['order_id'].nunique(),
        'BI_FakeFail_order_count': len(set(x[x['result']=='fake_fail']['order_id'])),

        'real_FF_orders': len(set(x[(x['final_result']==1)]['order_id']))
        }
    return pd.Series(names)

# Phase 4: get vol_of_ontime_KPI
def get_hub_info():
    # hub info

  auth.authenticate_user()

  creds, _ = default()

  gc = gspread.authorize(creds)
  hub_info_worksheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/19X4i0LxQPfP7dhp1hFVRBD_b8PQzKvqV4hRA01iThT4/edit?usp=sharing').worksheet("hub_info")

  # get_all_values gives a list of rows.

  hub_info = get_as_dataframe(hub_info_worksheet).dropna(axis=1, how='all').dropna(axis=0, how='all')
  return hub_info

# Phase 5: mapping infor
def mapping_phase(x, url):
  x = x.groupby(['driver_name', 'driver_type','first_attempt_date', 'hub_id',  'hub_name',	'region']).apply(bi_agg).reset_index() ###
  x.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/driver_groupby_attempt_date.csv', index=False)
  hub_info = get_hub_info()
  agg = pd.read_csv(url)[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'volume_of_ontime_KPI' ]]
  agg.rename(columns={"volume_of_ontime_KPI": 'Total orders reach LM hub' }, inplace=True)

  agg_driver = x.merge(agg, how='left', left_on=['first_attempt_date','hub_id'], right_on=['dest_hub_date','dest_hub_id'],suffixes=('', '_y'))

  # agg_driver['Total orders reach LM hub'] = agg_driver.fillna(0)['Total orders reach LM hub'] + agg_driver.fillna(0)['Total orders reach LM hub_y']
  agg_driver.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)
  agg_driver = agg_driver.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])

  return agg_driver
  
# Phase 6: Computing
def compute_phase(x):
  import math
  raw_data = x.copy()
  raw_data['BI_tracking_id'] = raw_data['BI_tracking_id'].astype(str)
  raw_data['BI_tracking_id'] = raw_data['BI_tracking_id'].str[1:-1]
  max_total_order = raw_data.groupby(['hub_id', 'first_attempt_date'])[['Total orders reach LM hub']].transform(lambda x: x.max())
  raw_data['FF_index'] = raw_data['BI_FakeFail_order_count']/max_total_order['Total orders reach LM hub']
  raw_data.describe().transpose()
  return pd.DataFrame(raw_data)

# Final: exporting
def export_final_driver_file(final):
  final = final.sort_values('first_attempt_date')
  # raw_data.to_csv('/content/sample_data/final_driver_data.csv')
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/fianl_data_monthly/final_driver_data_'+ str(dt.datetime.now().month) + '_' + str((dt.datetime.now().date().year)) +'.csv', index = False)
  # dashboard final data
  try:
    final.drop(columns=['BI_tracking_id']).to_csv('/content/DB_final_driver_data_'+ str(dt.datetime.now().month) + '_' + str((dt.datetime.now().date().year)) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_driver_data_'+ str(dt.datetime.now().month) + '_' + str((dt.datetime.now().date().year)) +'.csv', index = False)

















def read_pipeline(url_agg):
  clear_output()
  # reading and preprecessing
  print('Phase 1: Reading Data and preprocessing' + '-'*100)
  df = pre_processing(pd.concat([read_folder_pod_resultQA_in_month().drop(columns=['final_result', 'disputing', 'corrected_dispute']), reading_last_7_day()], ignore_index=False))
  # dispute
  clear_output()
  print('Phase 2: Preprocessing, Disputing, and Groupby Driver counting' + '-'*100)
  df = final_dispute(df)
  spliting_file(df)
        # print
  print(df['attempt_date'].unique())
  print("Number of Unique Driver_name: ", df['driver_name'].nunique())
  print("Number of Unique Driver_type: ", df['driver_type'].value_counts())
  print("Number of Uni Hub name: ", df['hub_name'].nunique())
  print("Shape: ", df.shape)
  print(df.info())



  # pre-compute phase
  clear_output()
  print('Phase 3: Mapping' + '-'*100)
  driver = mapping_phase(df, url_agg)


  # compute phase
  clear_output()
  print('Final: ' + '-'*100)
  final = compute_phase(driver)
  export_final_driver_file(final)
  return final