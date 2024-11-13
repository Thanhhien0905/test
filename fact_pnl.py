import os
import paramiko
import modin.pandas as pd
import numpy as np
from get_data_vn import *
import psycopg2
import logging
from datetime import datetime
from psycopg2.extras import execute_values
import logging
from logging.handlers import TimedRotatingFileHandler
log_handler = TimedRotatingFileHandler(
    filename='/home/hien_ntt/LoadDataFinal/log/fact_pnl.txt',  # ÄÆ°á»ng dáº«n Ä‘áº¿n file log
    when='D',                            # Xoay vÃ²ng theo ngÃ y
    interval=10,                         # Xoay vÃ²ng sau 10 ngÃ y
    backupCount=0                        # KhÃ´ng giá»¯ láº¡i file log cÅ©
)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
def connect_to_db():
    conn = psycopg2.connect(
        host='192.168.110.169',
        dbname='FIN_PNL',
        port='5432',
        user='postgres',
        password='admin'
    )
    return conn

def resample(df, sample_duration, type_data):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    df = df.resample(f"{sample_duration}{type_data}", label='left').apply(ohlc_dict).dropna().reset_index()
    return df
def MDD(df): 
    return df['dd'].max() 
def Sharpe(df): 
    # gain2 duoc tinh  theo %
    df['gain2'] = df['gain']/df['Close'] * 100 
    gain2_std = df['gain2'].std()
    if gain2_std == 0 or np.isnan(gain2_std):
        return 0 
    else:
        return df['gain2'].mean() / gain2_std * np.sqrt(252)
def calculate_metrics(df):
    df_ps = get_data_ps(1000)
    df_ps = resample(df_ps, 1, 'D')
    df_ps['Datetime'] = df_ps['Date']
    df_ps = df_ps[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['total_gain_max'] = df['total_gain'].cummax() 
    df['dd'] = df['total_gain_max'] - df['total_gain']
    df = pd.merge(df, df_ps, on='Date', how='inner')
    # Hitrate per day 
    hit_rate_per_day = round((len(df[df['gain'] > 0]) / len(df)) * 100, 2)
    df['gain2'] = df['gain'] / df['Close'] * 100
    return_per_year = round(df['gain2'].mean() * np.sqrt(252), 2)
    sharp = round(Sharpe(df), 2)

    return {
        "hit_rate_per_day": hit_rate_per_day,
        "rounded_mdd_score": round(df['dd'].max(), 2) if 'dd' in df.columns else 0,
        "profit_after_fee": df['total_gain'].iloc[-1],
        "return_per_year": return_per_year,
        "sharp": sharp,
    }

def calculate_and_insert_pnl():
    conn = connect_to_db()
    cursor = conn.cursor()
    logging.info('Ket noi thanh cong')

    cursor.execute("SELECT alpha_key, author_key, gain, total_gain, date FROM fact_daily ORDER BY date")
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=['alpha_key', 'author_key', 'gain', 'total_gain', 'Date'])
    grouped = df.groupby(['alpha_key', 'author_key'])
    
    data_to_insert = []
    batch_size = 10
    for (alpha_key, author_key), group in grouped:
        try:
            max_date = group['Date'].max()
            current_date = datetime.now().date()

            # Ki?m tra n?u ngày hi?n t?i l?n hon max_date >= 10 ngày thì b? qua alpha này
            if (current_date - max_date).days >= 10:
                continue
            else: 
                metrics = calculate_metrics(group)
                data_to_insert.append((
                    alpha_key, 
                    author_key, 
                    0, 
                    float(metrics['rounded_mdd_score']),  
                    0, 
                    0, 
                    float(metrics['profit_after_fee']),
                    0, 
                    0, 
                    float(metrics['return_per_year']),  
                    0, 
                    0,
                    float(metrics['hit_rate_per_day']), 
                    0, 
                    0, 
                    float(metrics['sharp'])
                ))
                cursor.execute("DELETE FROM fact_pnl01 WHERE alpha_key = %s AND author_key = %s", (alpha_key, author_key))
                # N?u d? kích thu?c lô, th?c hi?n chèn
                if len(data_to_insert) >= batch_size:
                    execute_values(cursor, '''
                        INSERT INTO fact_pnl01 
                        (alpha_key, author_key, margin, mdd_score, total_trading_quantity, total_profit, profit_after_fee,
                        trading_quantity_per_day, profit_per_day_after_fee, return_per_year, profit_per_year, hitrate, 
                        hitrate_per_day, mdd_percent, sharp, sharp_after_fee)
                        VALUES %s ON CONFLICT (alpha_key, author_key) 
                    ''', data_to_insert)
                    conn.commit()  
                    data_to_insert = []  

        except Exception as e:
            logging.warning(f"Error processing {alpha_key}, {author_key}: {e}")

    if data_to_insert:
        execute_values(cursor, '''
            INSERT INTO fact_pnl01 
            (alpha_key, author_key, margin, mdd_score, total_trading_quantity, total_profit, profit_after_fee,
             trading_quantity_per_day, profit_per_day_after_fee, return_per_year, profit_per_year, hitrate, 
             hitrate_per_day, mdd_percent, sharp, sharp_after_fee)
            VALUES %s ON CONFLICT (alpha_key, author_key) 
        ''', data_to_insert)
        conn.commit()
    cursor.close()
    conn.close()

def main():
    try:
        calculate_and_insert_pnl()
        print("Thuc hien thanh cong")
    except Exception as e:
        logging.warning(f"Da co loi trong qua trinh xu ly: {e}")

if __name__ == "__main__":
    main()
