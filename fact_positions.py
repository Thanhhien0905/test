import os
import paramiko
import pandas as pd
import psycopg2
import logging
import logging
from logging.handlers import TimedRotatingFileHandler
log_handler = TimedRotatingFileHandler(
    filename='/home/hien_ntt/LoadDataFinal/log/fact_positions.txt',  # Đường dẫn đến file log
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

# Hàm để lấy author_key từ bảng dim_alpha dựa vào alpha_key
def get_author_key_from_alpha(cursor, alpha_key):
    try:
        query = "SELECT author_key FROM dim_alpha WHERE alpha_key = %s"
        cursor.execute(query, (alpha_key,))
        result = cursor.fetchone()
        
        if result:
            return result[0]  
        else:
            return None
    
    except Exception as e:
        logging.error(f"Error fetching author_key: {e}")
        return None

# Hàm xử lý và chèn dữ liệu từ CSV vào bảng fact_position01
def process_csv_in_nas(cursor, connection, nas):
    hostname = "192.168.110.25"
    port = 22
    username = "hien_ntt"
    password = "hien925"
    # Kết nối đến NAS bằng SSH
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port=port, username=username, password=password)
    sftp = client.open_sftp()
    remote_dir = nas  # Thư mục NAS được truyền vào từ tham số

    try:
        files = sftp.listdir(remote_dir)
        
        # Tìm file CSV mới nhất trong thư mục
        latest_file = None
        latest_mtime = 0
        
        for file in files:
            if file.endswith('.csv') and not file.endswith('.csv.bak'):
                file_path = os.path.join(remote_dir, file).replace('\\', '/')
                file_stat = sftp.stat(file_path)
                
                # Kiểm tra thời gian sửa đổi của file
                if file_stat.st_mtime > latest_mtime:
                    latest_mtime = file_stat.st_mtime
                    latest_file = file_path
        
        # Nếu tìm thấy file mới nhất
        if latest_file:
            logging.info(f"Processing latest file: {latest_file}")
            with sftp.file(latest_file, 'r') as f:
                df = pd.read_csv(f)
                df['Time'] = pd.to_datetime(df['Time'], format='%Y_%m_%d_%H:%M:%S.%f') 
                df['Name'] = df['Name'].str.replace('_CP.txt', '')  
                df.rename(columns={'Time': 'date', 'Name': 'alpha_key', 'Current_Pos': 'current_pos', 'Price': 'close'}, inplace=True)
                df['position'] = df['current_pos'].apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
                
                # Chèn dữ liệu vào bảng fact_positions01
                for _, row in df.iterrows():
                    author_key = get_author_key_from_alpha(cursor, row['alpha_key'])
                    if author_key is None:
                        continue  
                    cursor.execute("""
                        INSERT INTO fact_positions01 (date, author_key, alpha_key, position, close)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (row['date'], author_key, row['alpha_key'], row['position'], row['close']))
                connection.commit()
            logging.info("Data processed and inserted successfully.")
        else:
            logging.info("No suitable CSV files found.")
    except Exception as e:
        logging.error(f"Error processing files: {e}") 
    finally:
        sftp.close()
        client.close()

def main():
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        process_csv_in_nas(cursor,connection, nas="/home/hien_ntt/nas_hn/HUNG/log/Daily_Pos")
    except Exception as e:
        logging.error(f"Database error: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    main()
