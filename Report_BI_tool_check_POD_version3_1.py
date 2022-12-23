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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

drive.mount('/content/drive', force_remount=True)
pd.set_option('display.max_columns', None)
auth.authenticate_user()

# building plotly dashboard
import plotly.io as pio
pio.renderers.default = 'colab'

# print bar progress bar in colaboratory
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "") (Str)

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


# module:
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
  x.drop(columns=['Unnamed: 0',  'Cuộc gọi phải phát sinh trước 8PM', 'first_attempt_date'], inplace=True, errors='ignore')
  x = x.sort_values(['attempt_datetime'])
  x = x.dropna(how='all', axis=1).dropna(how='all', axis=0) 
  # x = x.drop_duplicates(subset=['order_id','waypoint_id'], keep='last')
  x.attempt_datetime = pd.to_datetime(x.attempt_datetime)
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
  x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
  x[['hub_id', 'order_id', 'waypoint_id']] = x[['hub_id', 'order_id', 'waypoint_id']].astype('int64')
  x['count_call_log'] = x['count_call_log'].fillna(0)
  x['driver_type'] =  x.driver_name.apply(driver_finder)
  x =  x.dropna(how='all', axis=1).dropna(how='all', axis=0)
  x = key_shipper(x)
  x = get_first_attempt_date(x)

  print('#end')
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

  for i in url:
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
    print('Done File: {}'.format(test['attempt_date'].unique()[0]))

    
      

def read_folder_pod_resultQA_in_month(str_time_from, str_time_to):
  # Get data file names
  mypath = '/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'
  source_df = pd.DataFrame({
    'filename': [mypath + f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))],
    'date' : pd.to_datetime(pd.Series([item.replace(".csv", "") for item in [f for f in listdir(mypath) if (isfile(join(mypath, f)) & (os.path.splitext(os.path.basename(f))[1] == '.csv'))]]))
  })
  needed_df = source_df.loc[ (source_df.date >= pd.Timestamp(str_time_from)) & (source_df.date <= pd.Timestamp(str_time_to))] # select continually update date range

  # get data frame
  dfs = []
  for filename in needed_df['filename']:
     
      # read file
      renamed = pd.read_csv(filename)
      # append to list
      dfs.append(renamed)
  

  
  # Concatenate all data into one DataFrame
  big_frame = pd.concat(dfs, ignore_index=True)
  big_frame.info()
  return big_frame

# Phase 2: pre-processing, dispute
def driver_finder(x):
    if 'NEXT' in x: return 'fulltime'
    elif 'FRLA' in x: return 'freelancer'
    elif( 'FTS' in x) | ('AGAR' in x) | ('189-FRLA' in x) | ('518-FRLA' in x) | ('XDOCT' in x) | ('TSS' in x) | ('GRAB' in x) | ('RAGA' in x) |('AHA' in x) : return '3PLs'
    else: return 'others'


def get_disputetime():
    return (dt.datetime.now() - dt.timedelta(days=5)).date()

def final_dispute(x):
  ## Disputing
  x['disputing'] = 0
  x['attempt_datetime'] = pd.to_datetime(x['attempt_datetime'])
  x['final_result'] = 0
  x['corrected_dispute'] = 0
  x['affected_by_mass_bug'] = 0
  x['affected_by_discreting_bug'] = 0
  url = [
    'https://docs.google.com/spreadsheets/d/1i1Rha9Qg1qZ9sGI0-ddX9QBlO6Jg9URy2tm62Fu3X20/edit#gid=1966091300',
    'https://docs.google.com/spreadsheets/d/14J704NXH8hKJ4XWmmoGWEKozA66lJhui1HIaSVsAacw/edit#gid=1409944538',
    'https://docs.google.com/spreadsheets/d/1tcc9oOKLX7u5iVksTXVaG421KEDSLDhrK2IB8BrS-4w/edit#gid=1518203630',
    'https://docs.google.com/spreadsheets/d/1P0ohdLCGGvk037IHEFeiGvvc7l2bku5HIYCSgLT4i4o/edit#gid=419800374',
    'https://docs.google.com/spreadsheets/d/1TG97G9-h5VwfnvLwTWPfK9myq41galjDvZmCcmPTE74/edit#gid=1171452712',
    'https://docs.google.com/spreadsheets/d/1ZU_M3haDS-r2bQ-Y2II4rSOhtdkV96DMEK1ua4BopY8/edit?usp=sharing'

  ]

  disputing = pd.DataFrame()
  for i in url:
    # holding temp data
    print(i)
    creds, _ = default()
    gc = gspread.authorize(creds)
    temp = gc.open_by_url(i).worksheet("Detail")
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
  # collecting tu form product:
  creds, _ = default()
  gc = gspread.authorize(creds)
  print('https://docs.google.com/spreadsheets/d/1TLprj6Z9eerZzhph1nf24hyrBz_ApRYHlXZpmGSauww/edit?usp=sharing')
  temp = gc.open_by_url('https://docs.google.com/spreadsheets/d/1TLprj6Z9eerZzhph1nf24hyrBz_ApRYHlXZpmGSauww/edit?usp=sharing').worksheet("Form Responses 1")
  tid_product_form = get_as_dataframe(temp, evaluate_formulas=True).dropna(how='all', axis=1).dropna(how='all', axis=0).drop_duplicates(subset=['Mã đơn hàng (TID)']).rename(columns={"Mã đơn hàng (TID)": 'tracking_id' })[['tracking_id', 'PDT confirm']]
  x.loc[(x['tracking_id'].isin(tid_product_form.loc[tid_product_form['PDT confirm'] =='accept','tracking_id'])) & (x['tracking_id'].isin(x.loc[x['affected_by_mass_bug'] == 0,'tracking_id'])),'affected_by_discreting_bug'] = 1
  

  # disputing:
  x.loc[x['waypoint_id'].isin(disputing['waypoint_id']), 'disputing'] = 1


  # final:
  print('Bug case: ', x[ (x['fully_driver_result'] == 'fake_fail') & (x['affected_by_discreting_bug'] == 0) & ((x['affected_by_mass_bug'] == 0) & (x['corrected_dispute'] == 0)) ].shape)
  x.loc[ (x['fully_driver_result'] == 'fake_fail') & (x['affected_by_discreting_bug'] == 0) & ((x['affected_by_mass_bug'] == 0) & (x['corrected_dispute'] == 0))   , 'final_result'] = 1

  ## Nocode here
  print('Done Dispute!')
  return pd.DataFrame(x)

def spliting_file(x, split_from, split_to):
  x['attempt_date'] = pd.to_datetime(x['attempt_date']).dt.date
  for i in x.loc[(x['attempt_date'] >= pd.Timestamp(split_from)) & (x['attempt_date'] <= pd.Timestamp(split_to)), 'attempt_date'].unique():
    x[x['attempt_date'] == i].to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/Report BI Tool/Pre_processed data/'+str(i)+'.csv', index=False)
    print("Done file: " + str(i))
  print('DONE SLITING')


def sales_channel(x):
  print(x.info())
  x.loc[(~x['sales_channel'].isna()) & (x['fully_driver_result']=='fake_fail')].groupby(['first_attempt_date','region', 'sales_channel'])['tracking_id'].nunique().reset_index().to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/FF Oct Final/sales_channels_ffcase ' + str((pd.to_datetime(x['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(x['first_attempt_date']).dt.month.min()) +'_to_'+ str((pd.to_datetime(x['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(x['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  print('Done sales_channel~~~~~~~~~~~~~~')




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
        'final_disputation_attempt':  x[(x['corrected_dispute']==0) & (x['fully_driver_result']=='fake_fail')]['waypoint_id'].nunique(),


        ## tracking id
        'total_orders': x['order_id'].nunique(),
        'total_fake_fail_orders': x[x['fully_driver_result']=='fake_fail']['order_id'].nunique(),
        'disputing_orders':  x[(x['disputing']==1)]['order_id'].nunique(),
        'correted_by_disputing_orders':  x[(x['corrected_dispute']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'total_orders_affected_by_mass_bug': x[(x['affected_by_mass_bug']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'total_orders_affected_by_discreting_bug': x[(x['affected_by_discreting_bug']==1) & (x['fully_driver_result']=='fake_fail')]['order_id'].nunique(),
        'real_FF_orders': x[(x['final_result']==1)]['order_id'].nunique(),
        'Final_Fake_fail_tracking_id_list': x[(x['final_result']==1)]['tracking_id'].unique()
        }
    return pd.Series(names)

# Phase 5: grouping by driver_id
def mapping_phase(x, url):
  # load hub_info
  hub_info = pd.read_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/Dataset/Hubs enriched - hub_info.csv')
  
  # get volume_of_ontime_KPI
  volume_of_ontime_KPI = pd.read_csv(url)[['dest_hub_date', 'dest_hub_id', 'dest_hub_name', 'volume_of_ontime_KPI' ]]
  volume_of_ontime_KPI.rename(columns={"volume_of_ontime_KPI": 'Total orders reach LM hub' }, inplace=True)

  # group by driver_id apply bi_agg
  driver = x.groupby(['driver_id' ,'driver_name', 'driver_type','first_attempt_date', 'hub_id',  'hub_name',	'region']).apply(bi_agg).reset_index() ###

  # group by hub_id apply bi_agg
  hub = x.groupby(['first_attempt_date' ,'hub_id', 'hub_name', 'region']).apply(bi_agg).reset_index()

 
  # merge with volume_of_ontime_KPI
  agg_driver = driver.merge(volume_of_ontime_KPI, how='left', left_on=['first_attempt_date','hub_id'], right_on=['dest_hub_date','dest_hub_id'],suffixes=('', '_y'))
  agg_hub = hub.merge(volume_of_ontime_KPI, how='left', left_on=['first_attempt_date','hub_id'], right_on=['dest_hub_date','dest_hub_id'],suffixes=('', '_y'))

  # drop columns
  agg_driver.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)
  agg_hub.drop(columns=['dest_hub_date',	'dest_hub_id',	'dest_hub_name'], inplace=True)

  # merge with hub_info
  agg_driver = agg_driver.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])
  agg_hub = agg_hub.merge(hub_info, how='left', left_on=['hub_id'], right_on=['ID'],suffixes=('', '_')).drop(columns=['ID', 'Is Deleted', 'Name', 'Province Code', 'Region'])

  return agg_driver, agg_hub
  
# Phase 6: Computing
def compute_phase(x):
  import math
  raw_data = x.copy()
  max_total_order = raw_data.groupby(['hub_id', 'first_attempt_date'])[['Total orders reach LM hub']].transform(lambda x: x.max())
  raw_data['original_FF_index'] = (raw_data['total_fake_fail_orders']/max_total_order['Total orders reach LM hub']).fillna(0)
  raw_data['actual_fakefail_index'] = (raw_data['total_fake_fail_attempt']/raw_data['total_attempt']).fillna(0)

  return raw_data

# Final: exporting
def export_final_driver_file(final):
  # final driver data
  final = final.sort_values('first_attempt_date')

  # export final driver data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_driver_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)

def export_final_hub_file(final):
  # final hub data
  final = final.sort_values('first_attempt_date')

  # export final hub data
  final.to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  
  # dashboard final data
  try:
    final.drop(columns=['attempt_fake_fail_list', 'Final_Fake_fail_tracking_id_list']).to_csv('/content/DB_final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)
  except:
    final.to_csv('/content/DB_final_hub_data '+ str((pd.to_datetime(final['first_attempt_date']).dt.year.max())) + "_" + str(pd.to_datetime(final['first_attempt_date']).dt.month.max()) +'.csv', index = False)

def export_final_reason_file(x):
  print('Reason fail: start!')

  try:
    reason = pd.read_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/reason_fail_agg.csv')
  except:
    reason = pd.DataFrame()
  x = pd.concat([reason, x]).drop_duplicates(subset=['first_attempt_date', 'reason'],keep='last')
  x.sort_values('first_attempt_date').to_csv('/content/drive/MyDrive/VN-QA/29. QA - Data Analyst/FakeFail/final_data_monthly/reason_fail_agg.csv', index=False)
  print('Reason fail: end!')

  return x




### ___________________________________________________ Main ____________________________________________________________  
def read_pipeline(url_agg:str, str_time_from_:str, str_time_to_:str, split_from_:str, split_to_:str):
  print('hello Ninja !!!' + str(pd.Timestamp.now()))
  
  # Phase 1: Reading Data and preprocessing
  print('Phase 1: Reading Data and preprocessing' + '-'*100)
  big_frame = read_folder_pod_resultQA_in_month(str_time_from_, str_time_to_)
  df = pre_processing(big_frame.loc[(pd.to_datetime(big_frame['attempt_date']) >= pd.Timestamp(str_time_from_)) & (pd.to_datetime(big_frame['attempt_date']) <= pd.Timestamp(str_time_to_))])
  

  # Phase 2: Disputing, and Groupby Driver counting
  clear_output()
  sales_channel(df)
  print('Date collected: ', df['attempt_date'].unique())
  print('Phase 2: Disputing, and Groupby Driver counting' + '-'*100)
  df = final_dispute(df)
  spliting_file(df, split_from=split_from_, split_to=split_to_)
  



  # Phase 3: Mapping
  clear_output()
  print(df['attempt_date'].unique())
  print("Number of Unique Driver_name: ", df['driver_name'].nunique())
  print("Number of Unique Driver_type: ", df['driver_type'].value_counts())
  print("Number of Uni Hub name: ", df['hub_name'].nunique())
  print("Shape: ", df.shape)
  print(df.info())
  print('Phase 3: Mapping' + '-'*100)
  driver, hub = mapping_phase(df, url_agg)
  reason =  export_final_reason_file(df.groupby(['first_attempt_date', 'reason']).apply(reason_fail_agg).reset_index())
  
  
  # Phase 4: Aggregating
  clear_output()
  print('Final: ' + '-'*100)
  final_driver = compute_phase(driver)
  final_hub = compute_phase(hub)

  # Phase 5: Exporting
  export_final_driver_file(final_driver)
  export_final_hub_file(final_hub)
  return df, final_driver,reason