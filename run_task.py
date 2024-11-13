import subprocess
import time
from datetime import datetime

# Đường dẫn đến các file Python
base_path = "/home/hien_ntt/LoadDataFinal/"
dim_alpha_path = base_path + "dim_alpha.py"
dim_author_path = base_path + "dim_author.py"
fact_daily_path = base_path + "fact_daily.py"
fact_pnl_path = base_path + "fact_pnl.py"
fact_positions_path = base_path + "fact_positions.py"
fin_pnl_path = base_path + "FIN_PNL.py"
log_file_path = base_path + "error_log.txt"

# Hàm ghi log
def log_error(message):
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{datetime.now()} - {message}\n")

# Chạy dim_alpha và dim_author một lần lúc 9h sáng từ thứ Hai đến thứ Sáu
def run_daily_tasks():
    current_time = datetime.now()
    if current_time.weekday() < 5 and current_time.hour == 9 and current_time.minute == 0:
        try:
            subprocess.Popen(["python", dim_alpha_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log_error(f"Error running {dim_alpha_path}")
        
        try:
            subprocess.Popen(["python", dim_author_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            log_error(f"Error running {dim_author_path}")

# Chạy fact_daily và fact_pnl mỗi 30 phút từ 9h đến 16h từ thứ Hai đến thứ Sáu
def run_periodic_tasks():
    current_time = datetime.now()
    if current_time.weekday() < 5 and 9 <= current_time.hour <= 16:  # Từ 9h đến 16h
        try:
            subprocess.Popen(["python", fact_daily_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Chạy fact_daily
        except subprocess.CalledProcessError:
            log_error(f"Error running {fact_daily_path}")
        
        try:
            subprocess.Popen(["python", fact_pnl_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    # Sau đó chạy fact_pnl
        except subprocess.CalledProcessError:
            log_error(f"Error running {fact_pnl_path}")

# Chạy các file liên tục trong nền
def run_continuous_tasks():
    subprocess.Popen(["python", fact_positions_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen(["streamlit", "run", fin_pnl_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
run_continuous_tasks()

# Vòng lặp chính
while True:
    # Thực hiện nhiệm vụ hàng ngày
    run_daily_tasks()
    current_time = datetime.now()
    if current_time.weekday() < 5 and (current_time.hour >= 9 and current_time.minute in [0, 30]):
        run_periodic_tasks()
    if current_time.weekday() < 5 and current_time.hour == 16 and current_time.minute == 0:
        run_periodic_tasks()  # Chạy phiên cuối
        break  # Dừng vòng lặp
    time.sleep(60)
