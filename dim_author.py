import paramiko
import psycopg2
import logging

from logging.handlers import TimedRotatingFileHandler

# Cấu hình logging xoay vòng mỗi 10 ngày
log_handler = TimedRotatingFileHandler(
    filename='/home/hien_ntt/LoadDataFinal/log/dim_author.txt',  # Đường dẫn đến file log
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

# Hàm kiểm tra xem thư mục đã tồn tại trong bảng dim_author hay chưa
def folder_exists(cursor, folder_name):
    check_query = "SELECT COUNT(1) FROM dim_author WHERE author_key = %s"
    cursor.execute(check_query, (folder_name,))
    return cursor.fetchone()[0] > 0

# Hàm chèn tên folder vào bảng dim_author
def insert_folders_to_db(folders):
    conn = connect_to_db()
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO dim_author (author_key, role)
    VALUES (%s, %s)
    ON CONFLICT (author_key) DO NOTHING
    """ 

    for folder in folders:
        author_key = folder
        role = 'user'

        if not folder_exists(cursor, author_key):
            cursor.execute(insert_query, (author_key, role))
            logging.info(f'Inserted folder: {author_key}')  # Ghi log khi chèn thành công
        else:
            logging.info(f'Folder already exists: {author_key}')  # Ghi log khi thư mục đã tồn tại

    conn.commit()
    cursor.close()
    conn.close()

# Hàm lấy danh sách các thư mục từ NAS
def fetch_folders_from_nas():
    hostname = "192.168.110.25"
    port = 22
    username = "hien_ntt"
    password = "hien925"
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, port, username, password)
        sftp = client.open_sftp()

        nas_hcm = '/home/hien_ntt/nas_hcm/'
        nas_hn = '/home/hien_ntt/nas_hn/'

        hcm_folders = sftp.listdir(nas_hcm)
        hn_folders = sftp.listdir(nas_hn)

        return hcm_folders + hn_folders

    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()

# Hàm chính để kiểm tra và chèn thư mục mới
def check_and_insert_folders():
    logging.info("Checking for new folders...")
    folders = fetch_folders_from_nas()
    insert_folders_to_db(folders)
    logging.info("Inserted new folders into dim_author if any.")

# Chạy hàm chính
if __name__ == "__main__":
    check_and_insert_folders()
