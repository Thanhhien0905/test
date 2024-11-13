import os
import paramiko
import psycopg2
import logging
from logging.handlers import TimedRotatingFileHandler
# Cấu hình logging xoay vòng mỗi 10 ngày
log_handler = TimedRotatingFileHandler(
    filename='/home/hien_ntt/LoadDataFinal/log/dim_alpha.txt',  # Đường dẫn đến file log
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
    conn = psycopg2.connect(
        host='192.168.110.169',
        dbname='FIN_PNL',
        port='5432',
        user='postgres',
        password='admin'
    )
    return conn

# Hàm kiểm tra xem alpha_key đã tồn tại trong dim_alpha hay chưa
def alpha_key_exists(cursor, alpha_key):
    check_query = "SELECT COUNT(1) FROM dim_alpha WHERE alpha_key = %s"
    cursor.execute(check_query, (alpha_key,))
    return cursor.fetchone()[0] > 0

# Hàm xử lý các file PS_... .csv trong một thư mục
def process_csv_files_in_folder(cursor, folder):
    hostname = "192.168.110.25"
    port = 22
    username = "hien_ntt"
    password = "hien925"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, port, username, password)
        sftp = client.open_sftp()

        # Đường dẫn đến hai thư mục NAS
        nas_hcm_folder = f'/home/hien_ntt/nas_hcm/{folder}/'
        nas_hn_folder = f'/home/hien_ntt/nas_hn/{folder}/'

        # Kiểm tra và ghi lại danh sách thư mục có sẵn trên NAS HCM
        available_hcm_folders = sftp.listdir('/home/hien_ntt/nas_hcm/')
        logging.info(f"Available folders in NAS HCM: {available_hcm_folders}")

        # Xử lý thư mục NAS HCM
        if folder in available_hcm_folders:
            logging.info(f"Folder {nas_hcm_folder} exists.")
            csv_files_hcm = [f for f in sftp.listdir(nas_hcm_folder) if  (f.startswith('PS_') or f.startswith('PS2_') or f.startswith('CS_') or f.startswith('India_')) and f.endswith('.csv')]
            for csv_file in csv_files_hcm:
                alpha_key = os.path.splitext(csv_file)[0]  # Lấy alpha_key từ tên file
                if not alpha_key_exists(cursor, alpha_key):
                    insert_alpha_key_to_db(cursor, alpha_key, folder)
        else:
            logging.warning(f"Folder {nas_hcm_folder} does not exist.")

        # Kiểm tra và ghi lại danh sách thư mục có sẵn trên NAS HN
        available_hn_folders = sftp.listdir('/home/hien_ntt/nas_hn/')
        logging.info(f"Available folders in NAS HN: {available_hn_folders}")

        # Xử lý thư mục NAS HN
        if folder in available_hn_folders:
            logging.info(f"Folder {nas_hn_folder} exists.")
            # Kiểm tra sự tồn tại của thư mục trước khi liệt kê file
            try:
                csv_files_hn = [f for f in sftp.listdir(nas_hn_folder) if  (f.startswith('PS_') or f.startswith('PS2_') or f.startswith('CS_') or f.startswith('India_') or f.startswith('PS13h45_')) and f.endswith('.csv') and f != 'PS2_DAT_FBT.csv']
                for csv_file in csv_files_hn:
                    alpha_key = os.path.splitext(csv_file)[0]  # Lấy alpha_key từ tên file
                    if not alpha_key_exists(cursor, alpha_key):
                        insert_alpha_key_to_db(cursor, alpha_key, folder)
            except FileNotFoundError:
                logging.warning(f"Folder {nas_hn_folder} does not exist.")
        else:
            logging.warning(f"Folder {nas_hn_folder} does not exist.")

    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()


# Hàm chèn alpha_key vào bảng dim_alpha
def insert_alpha_key_to_db(cursor, alpha_key, author_key):
    insert_query = """
    INSERT INTO dim_alpha (alpha_key, author_key)
    VALUES (%s, %s)
    ON CONFLICT (alpha_key) DO NOTHING
    """
    try:
        cursor.execute(insert_query, (alpha_key, author_key))
        
    except Exception as e:
        logging.error(f'Failed to insert alpha_key: {alpha_key} for author_key: {author_key}. Error: {e}')

# Hàm lấy danh sách folders từ dim_author
def fetch_folders_from_db():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT author_key FROM dim_author")
    folders = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return folders

# Hàm chính
def check_and_insert_alpha_keys():
    conn = connect_to_db()
    cursor = conn.cursor()
    
    folders = fetch_folders_from_db() 
    for folder in folders:
        process_csv_files_in_folder(cursor, folder)

    conn.commit()
    cursor.close()
    conn.close()

# Chạy hàm chính
if __name__ == "__main__":
    check_and_insert_alpha_keys()
