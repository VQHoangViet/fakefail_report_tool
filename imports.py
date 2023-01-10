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
from dateutil.relativedelta import relativedelta
from google.auth import default
from google.colab import auth, drive
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from IPython.display import clear_output
from plotly.subplots import make_subplots

import requests

warnings.filterwarnings('ignore')





pd.set_option('display.max_columns', None)
auth.authenticate_user()
drive.mount('/content/drive', force_remount=True)