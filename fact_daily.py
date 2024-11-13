import paramiko
import psycopg2
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import  psycopg2.extras
import polars as pl
from pyspark.sql import functions as F
import logging
from logging.handlers import TimedRotatingFileHandler
log_handler = TimedRotatingFileHandler(
    filename='/home/hien_ntt/LoadDataFinal/log/fact_daily.txt',  # Đường dẫn đến file log
    when='D',                            # Xoay vòng theo ngày
    interval=10,                         # Xoay vòng sau 10 ngày
    backupCount=0                        # Không giữ lại file log cũ
)
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
# Hàm kết nối đến cơ sở dữ liệu PostgreSQL
def connect_to_db():
    return psycopg2.connect(
        host='192.168.110.169',
        dbname='FIN_PNL',
        port='5432',
        user='postgres',
        password='admin'
    )

# Hàm kiểm tra xem một alpha_key đã tồn tại trong fact_positions hay chưa
def last_date_for_alpha_key(cursor, alpha_key):
    cursor.execute("SELECT MAX(date) FROM fact_daily WHERE alpha_key = %s", (alpha_key,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else None

# Hàm chèn dữ liệu vào bảng fact_positions
def insert_daily_data(cursor, data):
    insert_query = """
    INSERT INTO fact_daily (author_key, alpha_key, date, gain, total_gain)
    VALUES %s
    ON CONFLICT (alpha_key, author_key, date) DO NOTHING
    """
    data_tuples = [tuple(row) for row in data[['author_key', 'alpha_key', 'date','gain', 'total_gain']].to_numpy()]
    # Chèn dữ liệu
    psycopg2.extras.execute_values(cursor, insert_query, data_tuples)
def all_dates_valid(date_series):
    if date_series.is_empty():
        return False
    try:
        date_series.str.strptime(pl.Date, strict=False)
        return True
    except Exception as e:
        logging.warning(f"Loi {e}")
        return False
    
def convert_date_format(date_str):
    if date_str is None:  # Kiểm tra nếu date_str là None
        return None
    for fmt in ('%m/%d/%Y', '%d/%m/%Y'):
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue  

    return date_str
def delete_old_data(cursor, alpha_key, last_date):
    delete_query = """
    DELETE FROM fact_daily 
    WHERE alpha_key = %s AND date = %s
    """
    cursor.execute(delete_query, (alpha_key, last_date))
# Hàm xử lý các file CSV trong thư mục NAS HCM
def process_csv_in_nas(cursor, nas):
    hostname = "192.168.110.25"
    port = 22
    username = "hien_ntt"
    password = "hien925"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, port, username, password)
        sftp = client.open_sftp()
        
        nas_folder_path = f'/home/hien_ntt/{nas}/'
        author_keys = [d for d in sftp.listdir(nas_folder_path) if is_sftp_dir(sftp, os.path.join(nas_folder_path, d))]

        for author_key in author_keys:
            author_folder_path = os.path.join(nas_folder_path, author_key)
            csv_files = [f for f in sftp.listdir(author_folder_path)]
            if not csv_files:
                logging.warning(f"Khong co file nao trong folder {author_folder_path}.")
                continue

            # Đọc toàn bộ file trong một lần
            csv_files_in_author_folder = sftp.listdir(author_folder_path)
            for daily_file in csv_files_in_author_folder:
                if (daily_file.startswith("PS") or daily_file.startswith("CS") or daily_file.startswith("India") or daily_file.startswith("PS2") or daily_file.startswith("PS13h45")) and daily_file.endswith(".csv") and daily_file != 'PS2_DAT_FBT.csv':
                    alpha_key = os.path.splitext(daily_file)[0] 
                    file_path = f"{author_folder_path}/{daily_file}"
                    file_stat = sftp.stat(file_path)
                    if file_stat.st_size == 0:  # Nếu file rỗng
                        logging.warning(f"File {daily_file} rong, bo qua.")
                        continue
                    with sftp.open(f"{author_folder_path}/{daily_file}") as file:
                        df = pl.read_csv(file)
                        columns = df.columns
                        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                        df = df.drop(unnamed_cols)
                       
                        date_col = next((col for col in df.columns if df[col].dtype == pl.String), None) 
                        gain_col = next((col for col in df.columns if col.lower() == 'gain'), None)
                        total_gain_col = next((col for col in df.columns if col.lower() == 'total_gain'), None)
                        if date_col is None or total_gain_col is None:
                            logging.warning(f"Thieu mot trong cac cot (date, total_gain) trong file: {daily_file}")
                            continue
                        if gain_col is None:
                            logging.warning(f"Cot gain thieu trong file: {daily_file}. Gan gia tri NA.")
                            df = df.with_columns(pl.lit(0).cast(pl.Float64).alias("gain"))
                            gain_col = "gain"
                           
                        df = df.filter(pl.col(date_col).is_not_null())
                        dates = df[date_col].to_list()  
                        converted_dates = [convert_date_format(date) for date in dates] 
                        df = df.with_columns(
                            pl.Series(date_col, converted_dates).alias(date_col)
                        )
            
                        if any(":" in date for date in dates):
                            logging.warning(f"File {daily_file} chua gia tri ngay voi gio")
                            df = df.with_columns(
                            pl.col(date_col).str.split(' ', inclusive=False).list.get(0).alias(date_col)
                        )

                        # Tiếp tục với chuyển đổi ngày tháng
                        try:
                            df = df.with_columns(
                                pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias('date')
                            )
                        except pl.exceptions.InvalidOperationError as e:
                            logging.warning(f"Loi chuyen doi ngay cho file {daily_file}: {e}")
                            continue

                        # Tiếp tục xử lý các cột gain và total_gain nếu dữ liệu hợp lệ
                        if not df['date'].is_null().any():
                            df = df.select([
                                pl.lit(alpha_key).alias('alpha_key'),
                                df['date'],
                                df[gain_col].alias('gain'),
                                df[total_gain_col].alias('total_gain')
                            ])

                            last_date = last_date_for_alpha_key(cursor, alpha_key)
                            delete_old_data(cursor, alpha_key, last_date)
                            if last_date is not None:
                                df = df.filter(pl.col('date') >= last_date)
                            
                            if not df.is_empty():
                                # Thêm author_key vào DataFrame và chèn vào cơ sở dữ liệu
                                df = df.with_columns(
                                    pl.lit(author_key).alias('author_key')
                                )
                                insert_daily_data(cursor, df)
                                cursor.connection.commit()
                        else:
                            logging.warning(f"Khong the chuyen doi gia tri trong cot date cua file {daily_file}.")
        if not author_keys:
            logging.warning("Khong tim thay tac gia nao")
    finally:
        client.close()

def is_sftp_dir(sftp, path):
    try:
        return sftp.stat(path).st_mode & 0o170000 == 0o40000
    except IOError:
        return False

def check_and_insert_positions():
    conn = connect_to_db()
    cursor = conn.cursor()
    process_csv_in_nas(cursor, 'nas_hcm')
    #truncate_ragged_lines=True
    process_csv_in_nas(cursor, 'nas_hn')
    #conn.commit()
    cursor.close()
    conn.close()

# Chạy hàm chính
if __name__ == "__main__":
    check_and_insert_positions()
