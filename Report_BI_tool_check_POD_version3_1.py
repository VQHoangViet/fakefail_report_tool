import datetime as dt
import glob
import os
import shutil
import warnings
from os import listdir
from os.path import isfile, join

import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from dateutil.relativedelta import relativedelta
from google.auth import default
from google.colab import auth, drive
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from IPython.display import clear_output
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')





pd.set_option('display.max_columns', None)
auth.authenticate_user()
drive.mount('/content/drive', force_remount=True)

# print bar progress bar in colaboratory
def printProgressBar (iteration, total, prefix = '', suffix = '', usepercent = True, decimals = 1, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        usepercent  - Optoinal  : display percentage (Bool)
        decimals    - Optional  : positive number of decimals in percent complete (Int), ignored if usepercent = False
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    # length is calculated by terminal width
    twx, twy = shutil.get_terminal_size()
    length = twx - 1 - len(prefix) - len(suffix) -4
    if usepercent:
        length = length - 6
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # process percent
    if usepercent:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='', flush=True)
    else:
        print('\r%s |%s| %s' % (prefix, bar, suffix), end='', flush=True)
    # Print New Line on Complete
    if iteration == total:
        print(flush=True)

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

  for j, i in enumerate(url):
    test = pd.read_csv('https://docs.google.com/spreadsheets/d/' + 
                    str(i.split("d/")[1].split("/e")[0]) +
                  '/export?gid=0&format=csv').drop_duplicates(subset=['order_id', 'waypoint_id', 'waypoint_photo_id'], keep='last')
    if pd.to_datetime(test['attempt_date'].unique()[0]) <= pd.Timestamp('2022-11-11'):
      test['no_call_log_aloninja'] = 0
      test['Không có hình ảnh POD'] = 0
      test.loc[test['count_call_log'] == 0, 'no_call_log_aloninja'] = 1
      test.loc[test['pod_photo'] == 'no photo', 'Không có hình ảnh POD'] = 1
      test = test.rename(columns={'Thời gian ghi nhận fail attempt phải trước 10PM' : 'Fail attempt sau 10PM',
      'Lịch sử tối thiểu 3 cuộc gọi': 'Lịch sử tối thiểu 3 cuộc gọi ra',
      'Thời gian giữa mỗi cuộc gọi tối thiểu 1 phút' : 'Thời gian giữa mỗi cuộc gọi tối thiểu 1p',
      'Thời gian đổ chuông >10s trong trường hợp khách không nghe máy' : 'Tối thiểu 3 cuộc gọi với thời gian đổ chuông >10s trong trường hợp khách không nghe máy',
      'no_call_log_aloninja' : 'Không có cuộc gọi tiêu chuẩn',
      'No Record' : 'Không có cuộc gọi thành công'}, errors='ignore').dropna(how='all', axis=1).drop(columns=['Unnamed: 0','Cuộc gọi phải phát sinh trước 8PM'], errors='ignore')
    test.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/{}.csv'.format(test['attempt_date'].unique()[0]), index=False)
    printProgressBar(j + 1, len(url), prefix = 'Progress:', suffix = 'Complete')
    print('Done File: {}\n'.format(test['attempt_date'].unique()[0]))

def read_folder_pod_resultQA_in_month(str_time_from, str_time_to):
  # Get data file names
  mypath = '/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'
  source_df = pd.DataFrame({
    'filename': [mypath + f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))],
    'date' : pd.to_datetime(pd.Series([item.replace(".csv", "") for item in [f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))]]))
  })
  needed_df = source_df.loc[ (source_df.date >= pd.Timestamp(str_time_from)) & (source_df.date <= pd.Timestamp(str_time_to))] # select continually update date rang
  # get data frame
  dfs = []
  for i, filename in enumerate(needed_df['filename']):
    printProgressBar(i + 1, len(needed_df['filename'].tolist()), prefix = 'Progress:', suffix = 'Complete')
    # append to list
    dfs.append(pd.read_csv(filename))

    # print progressBar to see the progress
  # Concatenate all data into one DataFrame
  big_frame = pd.concat(dfs, ignore_index=True)
  big_frame.info()
  return big_frame

def driver_finder(x):
    if 'NEXT' in x.upper(): return 'fulltime' 
    elif('AGAR' in x.upper()) | (
          '518-FRLA' in x.upper()) | (
            'XDOCT' in x.upper()) | (
              'XDOCK' in x.upper()) | (
                '936-TSS' in x.upper()) | (
                  'GRAB' in x.upper()) : return '3PLs'
    elif 'FRLA' in x.upper(): return 'freelancer'
    else: return 'others'

def get_first_attempt_date(x):
  try:
    x = x.drop(columns='first_attempt_date')
  except:
    pass
  first_attempt_date = x.groupby('tracking_id')['attempt_date'].min().reset_index().rename(columns={'attempt_date': 'first_attempt_date'})
  x = x.merge(first_attempt_date, how='inner', left_on='tracking_id', right_on='tracking_id')
  return x

def key_shipper(x):
  x.loc[x['tracking_id'].str.contains('KNJVN'), 'sales_channel'] = 'Tiktok'
  x.loc[x['tracking_id'].str.contains('NLVN'), 'sales_channel'] = 'Lazada'
  x.loc[(x['tracking_id'].str.contains('SPE')) | (x['sales_channel'].str.contains('RSPVNSPEVN')), 'sales_channel'] = 'Shopee'
  return x

def pre_processing(x):
  print('Preprocessing:...')

  # drop columns
  print('Drop columns:...')
  x.drop(columns=['Unnamed: 0',  'Cuộc gọi phải phát sinh trước 8PM', 'first_attempt_date'], inplace=True, errors='ignore')
  
  # convert to datetime
  print('Convert to datetime:...')
  x.attempt_datetime = pd.to_datetime(x.attempt_datetime)

  # sort by attempt_datetime
  print('Sort by attempt_datetime:...')
  x = x.sort_values(['attempt_datetime'])

  # drop na
  print('Drop na:...')
  x = x.dropna(how='all', axis=1).dropna(how='all', axis=0) 
  
  # conver int to float
  print('Convert int to float:...')
  x[['Fail attempt sau 10PM',
      'Lịch sử tối thiểu 3 cuộc gọi ra',
      'Thời gian giữa mỗi cuộc gọi tối thiểu 1p',
      'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại',
      'Tối thiểu 3 cuộc gọi với thời gian đổ chuông >10s trong trường hợp khách không nghe máy',
      'Không có cuộc gọi thành công',
      'Không có cuộc gọi tiêu chuẩn',
      'Không có hình ảnh POD']] = x[['Fail attempt sau 10PM',
      'Lịch sử tối thiểu 3 cuộc gọi ra',
      'Thời gian giữa mỗi cuộc gọi tối thiểu 1p',
      'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại',
      'Tối thiểu 3 cuộc gọi với thời gian đổ chuông >10s trong trường hợp khách không nghe máy',
      'Không có cuộc gọi thành công',
      'Không có cuộc gọi tiêu chuẩn',
      'Không có hình ảnh POD']].replace('-', 0).astype('float64')

  
  # convert to int
  print('Convert to int:...')
  x[['hub_id', 'order_id', 'waypoint_id']] = x[['hub_id', 'order_id', 'waypoint_id']].astype('int64')
  
  # fillna
  print('Fillna:...')
  x['count_call_log'] = x['count_call_log'].fillna(0)
  
  # driver type finder
  print('Driver type finder:...')
  x['driver_type'] =  x.driver_name.apply(driver_finder)

  # if fully_driver_result not exist, then create it and if it nan then fill it with result

  print('If fully_driver_result not exist, then create it:...')
  if 'fully_driver_result' not in x.columns:
    x['fully_driver_result'] = x['result']
  print('If fully_driver_result nan then fill it with result:...')
  x['fully_driver_result'] = x['fully_driver_result'].fillna(x['result'])
    # 3PLs fullly driver result = 'need_to_check'
  print('3PLs fullly driver result = need_to_check:...')
  x.loc[x['driver_type'] == '3PLs', 'fully_driver_result'] = 'need_to_check'

  
  # key shipper
  print('Key shipper:...')
  x = key_shipper(x)

  # first attempt date
  print('First attempt date:...')
  x = get_first_attempt_date(x)

  print('#end')
  return x

def read_POD_manual_check(x):

  # read list of POD gsheet url file
  creds, _ = default()
  gc = gspread.authorize(creds)
  temp = gc.open_by_url('https://docs.google.com/spreadsheets/d/1_i8lKyYwRKE05QlIWqyv83vld6J4LRb8osfSM5_qL38/edit#gid=0').worksheet("Sheet1")
  temp = get_as_dataframe(temp, evaluate_formulas=True)
  # read link in file
  url = temp['url'].dropna().unique()

  # read file
  df = pd.DataFrame()
  for i in url:
    print(i)
    temp = gc.open_by_url(i).worksheet("New Data Detail")
    temp = get_as_dataframe(temp, evaluate_formulas=True)[['Waypoint ID', 'FinalQualified/ Unqualified2']].dropna(how='all', axis=1).dropna(how='all', axis=0)
    df = pd.concat([df, temp], ignore_index=True)

  # create POD_sample_flag column
  x['POD_sample_flag'] = 0

  # create Final Unqualified POD column
  x['Final_Unqualified_POD_sample'] = 0
  
  # set POD_sample = 1 if waypoint_id in list
  x.loc[x['waypoint_id'].isin(df['Waypoint ID'].unique()), 'POD_sample_flag'] = 1

  # set POD_sample = 1 if FinalQualified/ Unqualified2 = Unqualified
  x.loc[x.waypoint_id.isin(df.loc[df['FinalQualified/ Unqualified2'] == 'Unqualified', 'Waypoint ID'].unique()), 'Final_Unqualified_POD_sample'] = 1

  return x

def dispute_phase(x):
  print('Final dispute:...') 
  

  ## Disputing
  x['disputing'] = 0
  x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
  x['corrected_dispute'] = 0
  x['affected_by_mass_bug'] = 0
  x['affected_by_discreting_bug'] = 0
  
  # read dispute backup
  creds, _ = default()
  gc = gspread.authorize(creds)
  temp = gc.open_by_url('https://docs.google.com/spreadsheets/d/1dTSoo5Pdf4Xhzhjdca1TXi4MYTfQ8gJAnUV7OAPblHc/edit#gid=0').worksheet("Sheet1")
  temp = get_as_dataframe(temp, evaluate_formulas=True)
  # read link in file
  url = temp['url'].dropna().unique()

  # read file
  disputing = pd.DataFrame()
  for j , i in enumerate(url):
    # holding temp data
    creds, _ = default()
    gc = gspread.authorize(creds)
    temp = gc.open_by_url(i).worksheet("Detail")
    printProgressBar(j + 1, len(url), prefix = 'Progress:', suffix = 'Complete link:' + i)
    # Convert to a DataFrame and render.
    disputing = pd.concat([disputing, get_as_dataframe(temp, evaluate_formulas=True)[['waypoint_id', 'Status']]])
  


  disputing =  disputing.dropna(how='all', axis=1).dropna(how='all', axis=0).drop_duplicates(subset=['waypoint_id'])
  accepted_waypoint = disputing[disputing['Status'].isin(['Corrected', 'Product xin loại trừ', 'no'])]
  
  x.loc[x['waypoint_id'].isin(accepted_waypoint['waypoint_id']), 'corrected_dispute'] = 1


  ### Mass bug
  # Lỗi sever 
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
  x = x.drop(columns=['callee_', 'driver_contact_'])
  # collecting from form product:
  creds, _ = default()
  gc = gspread.authorize(creds)
  print('https://docs.google.com/spreadsheets/d/1TLprj6Z9eerZzhph1nf24hyrBz_ApRYHlXZpmGSauww/edit?usp=sharing')
  temp = gc.open_by_url('https://docs.google.com/spreadsheets/d/1TLprj6Z9eerZzhph1nf24hyrBz_ApRYHlXZpmGSauww/edit?usp=sharing').worksheet("Form Responses 1")
  tid_product_form = get_as_dataframe(temp, evaluate_formulas=True).dropna(how='all', axis=1).dropna(how='all', axis=0).drop_duplicates(subset=['Mã đơn hàng (TID)']).rename(columns={"Mã đơn hàng (TID)": 'tracking_id' })[['tracking_id', 'PDT confirm']]
  x.loc[(x['tracking_id'].isin(tid_product_form.loc[tid_product_form['PDT confirm'] =='accept','tracking_id'])) & (x['tracking_id'].isin(x.loc[x['affected_by_mass_bug'] == 0,'tracking_id'])),'affected_by_discreting_bug'] = 1
  
  # collecting từ from product switch by condition 
  

  # disputing:
  x.loc[x['waypoint_id'].isin(disputing['waypoint_id']), 'disputing'] = 1
  ## Nocode here
  print('Done Dispute!')
  return pd.DataFrame(x)

def final_result(x):
    x['final_result'] = 0
    x['final_POD_result'] = 0
    # flag by bug or disputation
    x.loc[(x['fully_driver_result'] == 'fake_fail') & (x['affected_by_discreting_bug'] == 0) & ((x['affected_by_mass_bug'] == 0) & (x['corrected_dispute'] == 0))   , 'final_result'] = 1

    # flag by POD
    x.loc[(x['Final_Unqualified_POD_sample'] == 1) &  (x['affected_by_discreting_bug'] == 0) & (x['corrected_dispute'] == 0)  & (x['affected_by_discreting_bug'] == 0) & (x['affected_by_mass_bug'] == 0) , 'final_POD_result'] = 1
    return x

def spliting_file(x, split_from, split_to):
  x['attempt_date'] = pd.to_datetime(x['attempt_date']).dt.date
  for j, i in enumerate(x.loc[(x['attempt_date'] >= pd.Timestamp(split_from)) & (x['attempt_date'] <= pd.Timestamp(split_to)), 'attempt_date'].unique()):
    x[x['attempt_date'] == i].to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'+str(i)+'.csv', index=False)
    printProgressBar(j+1, len(x.loc[(x['attempt_date'] >= pd.Timestamp(split_from)) & (x['attempt_date'] <= pd.Timestamp(split_to)), 'attempt_date'].unique()), prefix = 'Progress:', suffix = 'Complete file: ' + str(i))


def sales_channel_for_OPEX(x):
  # sales_channel
  print('Start sales_channel...')
  x.loc[(~x['sales_channel'].isna()) & (x['fully_driver_result']=='fake_fail')] \
    .groupby(['first_attempt_date','region', 'sales_channel'])['tracking_id']\
      .nunique().\
        reset_index().\
          to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/FF Oct Final/sales_channels_ffcase ' + str((pd.to_datetime(x['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(x['first_attempt_date']).dt.month.min()) +'_to_'+ str((pd.to_datetime(x['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(x['first_attempt_date']).dt.month.max()) +'.csv', index = False)




# Phase 3: grouping
def reason_fail_agg(x):
    names = {
        'total_attempt': x['waypoint_id'].nunique(),
        'BI_FakeFail': x[x['fully_driver_result']=='fake_fail']['waypoint_id'].nunique(),
        'BI_FakeFail_order_count': x[x['fully_driver_result']=='fake_fail']['order_id'].nunique(),

        ## Updated on 27/09/2022 (starting time on data is 19/09/2022)
        'freelancer_FF_attempt' : x[(x['driver_type']=='freelancer') & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        'fulltime_FF_attempt' : x[(x['driver_type']=='fulltime') & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        '3PL_FF_attempt' : x[(x['driver_type']=='3PLs') & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        'others_FF_attempt' : x[(x['driver_type']=='others') & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        ##
        'freelancer_FF_orders' : x[(x['driver_type']=='freelancer') & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'fulltime_FF_orders' : x[(x['driver_type']=='fulltime') & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        '3PL_FF_orders' : x[(x['driver_type']=='3PLs') & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'others_FF_orders' : x[(x['driver_type']=='others') & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        ##

        # dispute case
        'dispute_case': x[x['disputing']==1]['waypoint_id'].nunique(),
       
        # final fully_driver_result
        'final_result': x[x['final_result']==1]['waypoint_id'].nunique(),
        
    }
    return pd.Series(names)


def bi_agg(x):
    names = {
        # note: count by attempt         # 'Cuộc gọi phải phát sinh trước 8PM': x[x['Cuộc gọi phải phát sinh trước 8PM']==1]['waypoint_id'].count(), (đã bị loại bỏ vào ngày 14/11/2022)
        'Fail attempt sau 10PM': x[(x['Fail attempt sau 10PM']==1)]['waypoint_id'].nunique(),
        'Lịch sử tối thiểu 3 cuộc gọi ra': x[x['Lịch sử tối thiểu 3 cuộc gọi ra']==1]['waypoint_id'].nunique(),
        'Tối thiểu 3 cuộc gọi với thời gian đổ chuông >10s trong trường hợp khách không nghe máy': x[x['Tối thiểu 3 cuộc gọi với thời gian đổ chuông >10s trong trường hợp khách không nghe máy']==1]['waypoint_id'].nunique(),
        'Thời gian giữa mỗi cuộc gọi tối thiểu 1p': x[x['Thời gian giữa mỗi cuộc gọi tối thiểu 1p']==1]['waypoint_id'].nunique(),
        'Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại': x[x['Thời gian gọi sớm hơn hoặc bằng thời gian xử lý thất bại']==1]['waypoint_id'].nunique(),
        'Không có cuộc gọi tiêu chuẩn': x[(x['Không có cuộc gọi tiêu chuẩn']==1)]['waypoint_id'].nunique(), 
        'Không có cuộc gọi thành công': x[x['Không có cuộc gọi thành công']==1]['waypoint_id'].nunique(),
        'Không có hình ảnh POD': x[x['Không có hình ảnh POD']==1]['waypoint_id'].nunique(),


        ## waypoint
        'total_attempt': x['waypoint_id'].nunique(),
        'total_fake_fail_attempt': x.loc[x['fully_driver_result'] == 'fake_fail','waypoint_id'].nunique(),
        'attempt_fake_fail_list': set(x[x['fully_driver_result']=='fake_fail']['waypoint_id']),
        'disputing_attempt':  x[(x['disputing']==1)]['waypoint_id'].nunique(),
        'correted_by_disputing_attempt':  x[(x['corrected_dispute']==0) & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        'total_attempt_affected_by_mass_bug': x[(x['affected_by_mass_bug']==1) & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        'total_attempt_affected_by_discreting_bug': x[(x['affected_by_discreting_bug']==1) & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),
        'total_POD_sample_attempt_flag': x[x['POD_sample_flag']==1]['waypoint_id'].nunique(),
        'total_POD_sample_attempt_flag_fake_fail': x[(x['Final_Unqualified_POD_sample']==1)]['waypoint_id'].nunique(),

        ## tracking id
        'total_orders': x['order_id'].nunique(),
        'total_fake_fail_orders': x[x['fully_driver_result']=='fake_fail']['order_id'].nunique(),
        'disputing_orders':  x[(x['disputing']==1)]['order_id'].nunique(),
        'correted_by_disputing_orders':  x[(x['corrected_dispute']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'total_orders_affected_by_mass_bug': x[(x['affected_by_mass_bug']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'total_orders_affected_by_discreting_bug': x[(x['affected_by_discreting_bug']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'total_POD_sample_orders_flag': x[x['POD_sample_flag']==1]['order_id'].nunique(),
        'total_POD_sample_orders_flag_fake_fail': x[(x['final_POD_result']==1)]['order_id'].nunique(),
        'real_FF_orders': x[(x['final_result']==1)]['order_id'].nunique(),
        'Final_Fake_fail_tracking_id_list': x[(x['final_result']==1)]['tracking_id'].unique()
        
        }
    return pd.Series(names)

# Phase 5: grouping by driver_id
def mapping_phase(x, url=''):
  # load hub_info
  hub_info = pd.read_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/Dataset/Hubs enriched - hub_info.csv')
  
  
  # get volume_of_ontime_KPI
  # volume_of_ontime_KPI_for_sales_channel =  pd.read_csv(url)[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'sales_channel', 'volume_of_ontime_KPI' ]]
  # volume_of_ontime_KPI_for_sales_channel['volume_of_ontime_KPI'] = volume_of_ontime_KPI_for_sales_channel['volume_of_ontime_KPI']/2
  # volume_of_ontime_KPI_for_sales_channel.rename(columns={"volume_of_ontime_KPI": 'Total orders reach LM hub' }, inplace=True)

  backup_volume_of_ontime_KPI = pd.read_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/volume_of_ontime_KPI/BACKUP_volume_of_ontime_kpi.csv')[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'volume_of_ontime_KPI' ]]
  if url == '':
    opex_ = pd.read_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/volume_of_ontime_KPI/Volume_reach_LM_hub 2022_12.csv')[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'volume_of_ontime_KPI' ]]
  else:
    opex_ = pd.read_csv(url)[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'volume_of_ontime_KPI' ]]
  volume_of_ontime_KPI = pd.concat([backup_volume_of_ontime_KPI, opex_]).drop_duplicates(subset=['dest_hub_date', 'dest_hub_id'], keep='last')
  # print max date
  print(volume_of_ontime_KPI.dest_hub_date.max())
  # save backup volume
  volume_of_ontime_KPI.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/volume_of_ontime_KPI/BACKUP_volume_of_ontime_kpi.csv', index=False)

  volume_of_ontime_KPI.rename(columns={"volume_of_ontime_KPI": 'Total orders reach LM hub' }, inplace=True, errors='ignore')
   # get min max first attempt date of x to slicing for volume_of_ontime_KPI 
  min_date = pd.to_datetime(x['first_attempt_date']).dt.date.min()
  max_date = pd.to_datetime(x['first_attempt_date']).dt.date.max()
  
  # slicing volume_of_ontime_KPI by min max date of x
  volume_of_ontime_KPI = volume_of_ontime_KPI[(pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.date >= min_date) & (pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.date <= max_date)]

  # to csv by min max time of volume_of_ontime_KPI dataframe
  volume_of_ontime_KPI.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/volume_of_ontime_KPI/volume_of_ontime_KPI '+ \
                              str((pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.year.min())) + "_" + str(pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.month.min()) + " " +\
                                str((pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.year.max())) + "_" + str(pd.to_datetime(volume_of_ontime_KPI['dest_hub_date']).dt.month.max()) +'.csv', index = False)

  # group by driver_id apply bi_agg
  driver = x.groupby(['driver_id' ,'driver_name', 'driver_type','first_attempt_date', 'hub_id',  'hub_name',	'region']).apply(bi_agg).reset_index() ###

  # group by hub_id apply bi_agg
  hub = x.groupby(['first_attempt_date' ,'hub_id', 'hub_name', 'region']).apply(bi_agg).reset_index()

  # group by sales_channel apply bi_agg
  # sales_channel = x.groupby(['first_attempt_date' ,'hub_id', 'hub_name','sales_channel']).apply(bi_agg).reset_index()
 
  # merge with volume_of_ontime_KPI
  agg_driver = driver.merge(volume_of_ontime_KPI, how='left', left_on=['first_attempt_date','hub_id'], right_on=['dest_hub_date','dest_hub_id'],suffixes=('', '_y'))
  agg_hub = hub.merge(volume_of_ontime_KPI, how='left', left_on=['first_attempt_date','hub_id'], right_on=['dest_hub_date','dest_hub_id'],suffixes=('', '_y'))
  # agg_sales_channel = sales_channel.merge(volume_of_ontime_KPI_for_sales_channel, how='left', left_on=['first_attempt_date','hub_id', 'sales_channel'], right_on=['dest_hub_date','dest_hub_id', 'sales_channel'],suffixes=('', '_y'))

  # drop columns
  agg_driver.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)
  agg_hub.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)
  # agg_sales_channel.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)

  # merge with hub_info
  agg_driver = agg_driver.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])
  agg_hub = agg_hub.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])
  # agg_sales_channel = agg_sales_channel.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])

  # GAP calculate volume_of_ontime_KPI and volume_of_ontime_KPI_Agg_hub
  gap = volume_of_ontime_KPI['Total orders reach LM hub'].sum() - agg_hub['Total orders reach LM hub'].sum()
  print('GAP: ', gap)
  

  return agg_driver, agg_hub #, agg_sales_channel
  
# Phase 6: Computing
def compute_phase(x):
  import math
  raw_data = x.copy()
  max_total_order = raw_data.groupby(['hub_id', 'first_attempt_date'])[['Total orders reach LM hub']].transform(lambda x: x.max())
  raw_data['original_FF_index'] = (raw_data['total_fake_fail_orders']/max_total_order['Total orders reach LM hub']).fillna(0)
  raw_data['actual_fakefail_index'] = (raw_data['real_FF_orders']/raw_data['Total orders reach LM hub']).fillna(0)
  
  return raw_data


# Final: exporting
def export_final_driver_file(final):
  # final driver data
  final = final.sort_values('first_attempt_date')

  # export final driver data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_driver/final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)

def export_final_hub_file(final):
  # final hub data
  final = final.sort_values('first_attempt_date')

  # export final hub data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_hub/final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)

def export_final_reason_file(x):
  # final reason data
  final = x.sort_values('first_attempt_date')

  # export final hub data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_reason/final_reason_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime( final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_reason_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    pass

def export_final_sales_channel_file(final):
  # final sales channel data
  final = final.sort_values('first_attempt_date')

  # export final sales channel data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_sales_channel/final_sales_channel_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_sales_channel_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_sales_channel_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)




### ___________________________________________________ Main ____________________________________________________________  
def read_pipeline(str_time_from_:str, str_time_to_:str , split_from_:str, split_to_:str, url_agg =''):
  
  print('hello Ninja !!!' + str(pd.Timestamp.now()))
  print('''
██╗  ██╗ █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗ 
██║  ██║██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗
███████║███████║███████╗   ██║   ███████║█████╗  ██████╔╝
██╔══██║██╔══██║╚════██║   ██║   ██╔══██║██╔══╝  ██╔══██╗
██║  ██║██║  ██║███████║   ██║   ██║  ██║███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
''')
  # Phase 1: Reading Data and preprocessing
  print('Phase 1: Reading Data and preprocessing' + '-'*100)
  big_frame = read_folder_pod_resultQA_in_month(str_time_from_, str_time_to_)
  df = pre_processing(big_frame.loc[(pd.to_datetime(big_frame['attempt_date']) >= pd.Timestamp(str_time_from_)) & (pd.to_datetime(big_frame['attempt_date']) <= pd.Timestamp(str_time_to_))])
  

  # Phase 2: Disputing, and Groupby Driver counting
  clear_output()
  print('Date collected: ', df['attempt_date'].unique())
  print('Phase 2: Disputing, and Groupby Driver counting' + '-'*100)
  df = read_POD_manual_check(df)
  df = dispute_phase(df)
  df = final_result(df)
  spliting_file(df, split_from=split_from_, split_to=split_to_)
  
  # Phase 3: Mapping
  clear_output()
  print('Phase 3: Mapping' + '-'*100)
  print("Number of Unique Driver_name: ", df['driver_name'].nunique())
  print("Number of Unique Driver_type: ", df['driver_type'].value_counts())
  print("Number of Uni Hub name: ", df['hub_name'].nunique())
  print("Shape: ", df.shape)
  print(df['fully_driver_result'].value_counts())
  driver, hub = mapping_phase(df, url_agg)
  
  
  # Phase 4: Aggregating
  clear_output()
  print('Phase 4: Aggregating' + '-'*100)
  print('Final: ' + '-'*100)
  final_driver = compute_phase(driver)
  final_hub = compute_phase(hub)
  

  # Phase 5: Exporting
  export_final_driver_file(final_driver)
  export_final_hub_file(final_hub)
  # export_final_sales_channel_file(sales_channel)
  reason =  export_final_reason_file(df.groupby(['first_attempt_date', 'reason']).apply(reason_fail_agg).reset_index())
  sales_channel_for_OPEX(df)

  return df, reason, final_driver, final_hub