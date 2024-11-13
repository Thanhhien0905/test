
import pandas as pd
import numpy as np
from time import sleep,mktime
from datetime import time,date,timedelta,datetime
import get_data_today, get_data_today_vn30
# from tqdm import tqdm
import requests
import os
# import time as ti

def get_data_ps(days):

    history_data_path = r"\\home\hien_ntt\nas_hn\HUY_PV\Library\Data\Historical_Data1.csv"
    start_date = date.today() - timedelta(days)
    # print(start_time)
    df1 = pd.read_csv(history_data_path, nrows= int(((datetime.now().date() - start_date).days+1) * 300))
    df2=get_data_today.get_data_ps(7)
    df = pd.concat([df1,df2])
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.date
    if len(df)>0:
        df = df.loc[df['day']>=start_date]
        df = df.sort_values('Date').groupby('Date').head(1).sort_values(by='Date',ascending=True)
        df.index = range(len(df))
        # print(df)
        return df

def get_data_vn30(days):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    history_data_path = os.path.join(current_directory, 'Historical_Data_Vn30.csv')  # Thay 'ten_file.csv' bằng tên thực tế của file CSV
    #today_data_path=os.path.join(current_directory, 'Today_Data_Vn30.csv')
    # start_time = ti.time()
    start_date = date.today() - timedelta(days)
    # print(start_time)
    df1 = pd.read_csv(history_data_path, nrows= int(((datetime.now().date() - start_date).days+1) * 300))
    df2=get_data_today_vn30.get_data_vn30(7)

    df = pd.concat([df1,df2])
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.date
    if len(df)>0:
        df = df.loc[df['day']>=start_date]
        df = df.sort_values('Date').groupby('Date').head(1).sort_values(by='Date',ascending=True)
        df.index = range(len(df))
        # print(df)
        return df
def get_data_ps_adjusted(days):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    history_data_path = os.path.join(current_directory, 'Historical_Data_Adjusted.csv')  # Thay 'ten_file.csv' bằng tên thực tế của file CSV
    start_date = date.today() - timedelta(days)
    # print(start_time)
    df1 = pd.read_csv(history_data_path, nrows= int(((datetime.now().date() - start_date).days+1) * 300))

    df2=get_data_today.get_data_ps(1)


    df = pd.concat([df1,df2])
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.date
    if len(df)>0:
        df = df.loc[df['day']>=start_date]
        df = df.sort_values('Date').groupby('Date').head(1).sort_values(by='Date',ascending=True)
        df.index = range(len(df))
        # print(df)
        return df

    
def trading_day():

    # Lấy thời gian hiện tại ở múi giờ địa phương
    today = datetime.now()

    list_day=['2024-01-01', '2024-02-08', '2024-02-09', '2024-02-12', '2024-02-13', '2024-02-14', '2024-04-18', 
            '2024-04-30', '2024-05-01','2024-09-02', '2024-09-03']

    list_day = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in list_day]

    if today.weekday()==5 or today.weekday()==6 or today.date() in list_day:
        return False
    else:
        return True