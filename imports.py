
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
