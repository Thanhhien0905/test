from datetime import datetime, time, timedelta
from matplotlib import ticker
from matplotlib.dates import DateFormatter
import mplcursors
import streamlit as st
import psycopg2
from psycopg2 import extras
import pandas as pd
from FDaSua import BacktestInformation, Sharp
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
from io import BytesIO
import numpy as np
st.set_page_config(layout="wide")
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import plotly.express as px
import plotly.graph_objects as go
import time
#HIHI123
# Hàm kết nối tới cơ sở dữ liệu PostgreSQL
def connect_to_db():
    conn = psycopg2.connect(
        host='192.168.110.169',
        dbname='FIN_PNL',
        port='5432',
        user='postgres',
        password='admin'
    )
    return conn
def authenticate(author_key, password):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT author_key, role FROM dim_author WHERE author_key = %s AND password = %s", (author_key, password))
    user_data = cur.fetchone()
    cur.close()
    conn.close()
    if user_data:
        return {"author_key": user_data[0],  "role": user_data[1]}
    return None

### Xem positions
# Hàm truy vấn lấy danh sách tác giả từ fact_positions01
def fetch_authors_by_fact_positions01(conn):
    query = "SELECT distinct author_key FROM fact_positions01 ORDER BY author_key;"
    with conn.cursor() as cur:
        cur.execute(query)
        authors = cur.fetchall()
    return [author[0] for author in authors]
# Hàm truy vấn lấy danh sách alpha theo tác giả từ bảng fact_positions01
def fetch_alphas_by_fact_positions01(conn, author_key):
    query = f"SELECT distinct f.alpha_key FROM fact_positions01 f WHERE f.author_key = %s ORDER BY f.alpha_key;"
    params = (author_key,)
    with conn.cursor() as cur:
        cur.execute(query,params)
        alphas = cur.fetchall()
    return [alpha[0] for alpha in alphas]
def fetch_position_by_author(conn, author_key,status_key):
    query = """
    WITH ranked_positions AS (
        SELECT f.*, 
               ROW_NUMBER() OVER (PARTITION BY f.alpha_key ORDER BY f.date DESC) AS rn
        FROM fact_positions01 f
        JOIN dim_alpha d ON f.alpha_key = d.alpha_key
        WHERE 1=1
    """

    # Thêm điều kiện lọc nếu có author_key và status_key
    params = ()
    if author_key and author_key != "Xem tất cả":
        query += " AND f.author_key = %s"
        params += (author_key,)

    if status_key and status_key != "Xem tất cả":
        query += " AND d.status = %s"
        params += (status_key,)

    # Đóng câu lệnh SQL
    query += """
    )
    SELECT f.alpha_key, f.author_key, f.date, f.position, f.close  -- Chỉ chọn các cột cần thiết
    FROM ranked_positions f
    WHERE rn = 1;
    """

    # Thực thi truy vấn và trả về dữ liệu
    with conn.cursor() as cur:
        cur.execute(query, params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()

    return colnames, pnl_data
# Hàm truy vấn lấy dữ liệu position theo alpha từ bảng fact_position và lấy tên cột
def fetch_position_by_alpha(conn, alpha_key):
    query = f"""
    WITH ranked_positions AS (
            SELECT f.*, 
                   ROW_NUMBER() OVER (PARTITION BY f.alpha_key ORDER BY f.date DESC) AS rn
            FROM fact_positions01 f
            WHERE f.alpha_key = %s
        )
        SELECT *
        FROM ranked_positions
        WHERE rn = 1;
    """
    params = (alpha_key,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()
    return colnames, pnl_data

def hien_thi_position_danh_muc(conn):
    # Lấy danh sách các danh mục từ bảng danhmuc_alpha
    danhmuc_data = fetch_danhmuc_alpha(conn)
    if not danhmuc_data:
        st.warning("Không có dữ liệu danh mục.")
        return
    
    # Tạo DataFrame từ dữ liệu danhmuc_alpha
    df_danhmuc = pd.DataFrame(danhmuc_data, columns=['list_author', 'list_name','list_date', 'alpha_key', 'trong_so'])
    # Nhóm theo list_author và list_name để tính vị thế tổng cho từng danh mục
    vithe_danh_muc = []
    for (list_author, list_name, list_date), group in df_danhmuc.groupby(['list_author', 'list_name', 'list_date']):
        alpha_keys = group['alpha_key'].tolist()
        trong_so_list = group['trong_so'].tolist()
        
        # Truy vấn vị thế (position) cho các alpha trong danh sách
        total_position = 0
        for alpha_key, trong_so in zip(alpha_keys, trong_so_list):
            colnames, position_data = fetch_position_by_alpha(conn, alpha_key)
            if position_data:
                position = position_data[0][colnames.index('position')]  # Giả sử cột vị thế là 'position'
                total_position += position * trong_so
        
        vithe_danh_muc.append({
            'list_author': list_author,
            'list_name': list_name,
            'list_date': list_date,    
            'total_position': total_position  # Vị thế tổng của danh mục
        })

    # Tạo DataFrame kết quả
    df_vithe_danh_muc = pd.DataFrame(vithe_danh_muc)
    
    # Tạo grid options cho AgGrid
    grid_options = GridOptionsBuilder.from_dataframe(df_vithe_danh_muc)
    grid_options.configure_selection('single')  # Chế độ chọn 1 hàng
    grid_options.configure_side_bar()  # Thêm thanh công cụ bên
    grid_options.configure_default_column(editable=True)  # Cho phép chỉnh sửa các cột
    grid_options = grid_options.build()
    
    # Hiển thị AgGrid
    grid_response = AgGrid(df_vithe_danh_muc, gridOptions=grid_options, fit_columns_on_grid_load=True, enable_enterprise_modules=True)
    st.write("Tổng của cột 'total_position':", df_vithe_danh_muc['total_position'].sum())


### Xem PNL
#Hàm hiển thị màu sắc
cellstyle_jscode = JsCode("""
    function(params) {
        if (params.data['sharp'] > 3.9 && params.data['hitrate']>=55) {
            return {
                'color': 'purple'
            };
        } else if (params.data['sharp'] > 3.5 && params.data['hitrate']>=55) {
            return {
                'color': 'pink'
            };
        } else if (params.data['sharp'] >= 3 && params.data['hitrate']>=48) {
            return {
                'color': 'green'
            };
        } else {
            return {
                'color': 'black'
            }
        }
    }
""")
# Hàm truy vấn lấy danh sách tác giả từ fact_pnl
def fetch_authors_by_fact_pnl(conn):
    query = "SELECT distinct author_key FROM fact_pnl01 ORDER BY author_key;"
    with conn.cursor() as cur:
        cur.execute(query)
        authors = cur.fetchall()
    return [author[0] for author in authors]


# Hàm truy vấn lấy dữ liệu PNL theo tác giả từ bảng fact_pnl và lấy tên cột
def fetch_pnl_by_author(conn, author_key, status=None):
    # Xây dựng câu lệnh SQL cơ bản
    query = """
    SELECT f.author_key, f.alpha_key, p.position, f.margin, f.profit_after_fee AS profit, 
        f.mdd_score, f.mdd_percent, f.trading_quantity_per_day AS trad_per_day, 
        f.sharp_after_fee AS sharp, f.return_per_year AS return, 
        f.hitrate, f.hitrate_per_day AS hit_per_day, d.type
    FROM fact_pnl01 f
    JOIN dim_alpha d ON d.alpha_key = f.alpha_key
    LEFT JOIN (
        SELECT p1.alpha_key, p1.position
        FROM (
            SELECT p1.alpha_key, p1.position, ROW_NUMBER() OVER (PARTITION BY p1.alpha_key ORDER BY p1.date DESC) AS rn
            FROM fact_positions01 p1
        ) p1
        WHERE p1.rn = 1
    ) p ON p.alpha_key = f.alpha_key
    WHERE 1=1 


    """
    # Thêm điều kiện lọc cho author_key nếu có
    if author_key and author_key != "Xem tất cả":
        query += " AND f.author_key = %s"
        params = (author_key,)
    else:
        params = ()

    # Thêm điều kiện lọc cho trạng thái nếu có
    if status and status != "Xem tất cả":
        if status == ' ':
            query += " AND (d.status is null or d.status = ' ')"
        else:
            query += " AND d.status = %s"
            params += (status,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()

    return colnames, pnl_data
# Lấy dữ liệu để vẽ đồ thị alpha chạy live
def fetch_data_plot_position_live(conn, author_key, alpha_key):
    query = """
    SELECT p.date, p.position, p.close
    FROM fact_positions01 p join fact_pnl01 f on p.alpha_key = f.alpha_key
    WHERE f.alpha_key = %s
    """
    params = [alpha_key]

    if author_key is not None:
        query += " AND f.author_key = %s"
        params.append(author_key)
    query += " ORDER BY p.date"
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
# Lấy dữ liệu để vẽ đồ thị
def fetch_data_plot_position(conn, author_key, alpha_key):
    query = """
    SELECT p.date, p.gain, p.total_gain
    FROM fact_daily p join fact_pnl01 f on p.alpha_key = f.alpha_key
    WHERE f.alpha_key = %s
    """
    params = [alpha_key]

    if author_key is not None:
        query += " AND f.author_key = %s"
        params.append(author_key)
    query += " ORDER BY p.date"
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()
# Hàm truy vấn lấy dữ liệu PNL theo alpha từ bảng fact_pnl và lấy tên cột
def fetch_pnl_by_alpha(conn, alpha_key):
    query = f"""
    SELECT f.author_key,f.alpha_key,f.margin,f.profit_after_fee as profit,f.mdd_score,f.mdd_percent,
    f.trading_quantity_per_day as trad_per_day, f.sharp_after_fee as sharp, f.return_per_year as return,
    f.hitrate , f.hitrate_per_day as hit_per_day
    FROM fact_pnl01 f
    WHERE f.alpha_key = %s;
    """
    params = [alpha_key]
    with conn.cursor() as cur:
        cur.execute(query,params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()
    return colnames, pnl_data
def check_alpha_in_fact_positions(conn, alpha_key):
    query = """
    SELECT COUNT(1) FROM fact_positions01 WHERE alpha_key = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (alpha_key,))
        result = cur.fetchone()
        return result[0] > 0
def plot_total_gain(conn, author_key, alpha_key):

    total_gain = fetch_data_plot_position(conn, author_key, alpha_key)
    total_gain = pd.DataFrame(total_gain, columns=['date', 'gain', 'total_gain'])
    total_gain['date'] = pd.to_datetime(total_gain['date'])
    total_gain.set_index('date', inplace=True)

    daily_total_gain = total_gain['total_gain'].resample("1D").last().dropna()

    # Tạo biểu đồ với Plotly
    fig = go.Figure()

    # Thêm dòng cho Total Gain
    fig.add_trace(go.Scatter(
        x=daily_total_gain.index,
        y=daily_total_gain.values,
        mode='lines',
        name='Total Gain',
        line=dict(color='#1f77b4')  # Màu xanh dương
    ))
    # Cập nhật bố cục của biểu đồ
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        width=1500,
        height=550,
        font=dict(family="Arial, sans-serif", size=14),
        template='plotly_dark',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.update_layout(
        shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1, xref='paper', yref='paper', line=dict(color="black", width=1))]
    )
    fig.update_xaxes(tickformat="%Y-%m-%d")
    st.plotly_chart(fig)
def pnl_mean(conn, author_key, status_key=None):
    colnames, pnl_data = fetch_pnl_by_author(conn, author_key, status_key)

    # Tính trung bình các thông số từ PnL data
    df_pnl = pd.DataFrame(pnl_data, columns=colnames)
    df_mean = df_pnl[['margin', 'profit', 'mdd_score', 'mdd_percent', 
                      'trad_per_day', 'sharp', 'return', 'hitrate', 'hit_per_day']].mean()
    df_mean = pd.DataFrame([df_mean.values], columns=df_mean.index)
    return df_mean


def fetch_data_all(conn, author_key, status=None):
    query = """
    SELECT fd.date, fd.total_gain
    FROM fact_daily fd
    JOIN fact_pnl01 f ON fd.alpha_key = f.alpha_key
    JOIN dim_alpha d ON d.alpha_key = f.alpha_key
    LEFT JOIN (
        SELECT p1.alpha_key, p1.position
        FROM (
            SELECT p1.alpha_key, p1.position, ROW_NUMBER() OVER (PARTITION BY p1.alpha_key ORDER BY p1.date DESC) AS rn
            FROM fact_positions01 p1
        ) p1
        WHERE p1.rn = 1
    ) p ON p.alpha_key = f.alpha_key
    WHERE 1=1
    """

    # Khởi tạo danh sách điều kiện và tham số
    params = []

    if author_key and author_key != "Xem tất cả":
        query += " AND f.author_key = %s"
        params = (author_key,)
    else:
        params = ()

    # Thêm điều kiện lọc cho trạng thái nếu có
    if status and status != "Xem tất cả":
        if status == ' ':
            query += " AND (d.status is null or d.status = ' ')"
        else:
            query += " AND d.status = %s"
            params += (status,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        # Tạo DataFrame từ kết quả truy vấn
        df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    return df

# Vẽ tất cả pnl 
def plot_all_total_gain(df):
    df['date'] = pd.to_datetime(df['date'])
    weighted_avg_per_day = df.groupby('date').apply(
        lambda x: (x['total_gain'].mean())
    )
    weighted_avg_df = weighted_avg_per_day.reset_index(name='weighted_avg')
    # Lọc bỏ các ngày có 'weighted_avg' bằng 0
    weighted_avg_df = weighted_avg_df[weighted_avg_df['weighted_avg'] != 0]
    # Vẽ biểu đồ với Plotly
    fig = px.line(weighted_avg_df, x='date', y='weighted_avg', title='PNL giả định')
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=18, family="Arial, sans-serif")
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        width=1500,
        height=550,
        font=dict(family="Arial, sans-serif", size=14),
        template='plotly_dark',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.update_layout(
        shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1, xref='paper', yref='paper', line=dict(color="black", width=1))]
    )
    fig.update_xaxes(tickformat="%Y-%m-%d")
    st.plotly_chart(fig)
# Hàm lấy danh sách danh mục từ bảng danhmuc_alpha
def fetch_danhmuc_alpha(conn):
    query = """
    SELECT list_author, list_name, list_date, alpha_key, trong_so 
    FROM danhmuc_alpha;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        danhmuc_data = cur.fetchall()
    return danhmuc_data

# Hàm lấy thông số PNL từ bảng fact_pnl theo alpha_key
def fetch_pnl_by_alpha_list(conn, alpha_keys):
    query = """
    SELECT f.alpha_key, f.margin, f.profit_after_fee as profit, f.mdd_score, f.mdd_percent, 
           f.trading_quantity_per_day as trad_per_day, f.sharp_after_fee as sharp, 
           f.return_per_year as return, f.hitrate, f.hitrate_per_day as hit_per_day
    FROM fact_pnl01 f
    WHERE f.alpha_key IN %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (tuple(alpha_keys),))
        pnl_data = cur.fetchall()
    return pnl_data

# Hàm tính toán và hiển thị bảng thông số danh mục với trung bình có trọng số
def hien_thi_thong_so_danh_muc(conn):
    # Lấy danh sách các danh mục từ bảng danhmuc_alpha
    danhmuc_data = fetch_danhmuc_alpha(conn)
    if danhmuc_data is None:
        return
    
    # Tạo DataFrame từ dữ liệu danhmuc_alpha
    df_danhmuc = pd.DataFrame(danhmuc_data, columns=['list_author', 'list_name','list_date', 'alpha_key', 'trong_so'])
    # Nhóm theo list_author và list_name để tính các thông số trung bình có trọng số
    thong_so_danh_muc = []
    for (list_author, list_name, list_date), group in df_danhmuc.groupby(['list_author', 'list_name', 'list_date']):
        alpha_keys = group['alpha_key'].tolist()
        trong_so_list = group['trong_so'].tolist()
        total_trong_so = sum(trong_so_list)  # Tổng trọng số của danh mục
        # Truy vấn thông số PNL cho danh sách alpha_keys
        pnl_data = fetch_pnl_by_alpha_list(conn, alpha_keys)
        df_pnl = pd.DataFrame(pnl_data, columns=['alpha_key', 'margin', 'profit', 'mdd_score', 
                                                 'mdd_percent', 'trad_per_day', 'sharp', 
                                                 'return', 'hitrate', 'hit_per_day'])
        df_combined = pd.merge(df_pnl, group[['alpha_key', 'trong_so']], on='alpha_key', how='inner')
        if df_combined.empty:
            
            continue
        # Tính toán các giá trị tổng hợp
        thong_so_tb = {
            'list_author': list_author,
            'list_name': list_name,
            'list_date': list_date,
            'margin': round((df_combined['margin'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'profit': round((df_combined['profit'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'mdd_score': round((df_combined['mdd_score'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'mdd_percent': round((df_combined['mdd_percent'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'trad_per_day': round((df_combined['trad_per_day'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'sharp': round((df_combined['sharp'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'return': round((df_combined['return'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'hitrate': round((df_combined['hitrate'] * df_combined['trong_so']).sum() / total_trong_so, 2),
            'hit_per_day': round((df_combined['hit_per_day'] * df_combined['trong_so']).sum() / total_trong_so, 2)
        }

        thong_so_danh_muc.append(thong_so_tb)

    # Tạo DataFrame kết quả
    df_thong_so_danh_muc = pd.DataFrame(thong_so_danh_muc)
    return df_thong_so_danh_muc
    
def fetch_pnl_list_name(conn, list_name):
    query = """
    SELECT a.alpha_key,a.author_key, p.position * a.trong_so as position, a.trong_so, f.margin,f.profit_after_fee as profit,f.mdd_score,f.mdd_percent,
    f.trading_quantity_per_day as trad_per_day, f.sharp_after_fee as sharp, f.return_per_year as return,
    f.hitrate , f.hitrate_per_day as hit_per_day
    FROM fact_pnl01 f
    JOIN danhmuc_alpha a ON a.alpha_key = f.alpha_key
    LEFT JOIN (
    SELECT p1.alpha_key, p1.position
    FROM fact_positions01 p1
    WHERE p1.date = (
        SELECT MAX(p2.date)
        FROM fact_positions01 p2
        WHERE p2.alpha_key = p1.alpha_key
    )
    ) p ON p.alpha_key = f.alpha_key
    WHERE a.list_name = %s
    """
    
    params = [list_name]
    with conn.cursor() as cur:
        cur.execute(query,params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()
    return colnames, pnl_data

def fetch_pnl_list_name_live(conn, list_name):
    query = """
    SELECT a.alpha_key,a.author_key, p.position * a.trong_so as position, a.trong_so
    FROM fact_pnl01 f
    JOIN danhmuc_alpha a ON a.alpha_key = f.alpha_key
    JOIN (
    SELECT p1.alpha_key, p1.position
    FROM fact_positions01 p1
    WHERE p1.date = (
        SELECT MAX(p2.date)
        FROM fact_positions01 p2
        WHERE p2.alpha_key = p1.alpha_key
    )
    ) p ON p.alpha_key = f.alpha_key
    WHERE a.list_name = %s
    """
    
    params = [list_name]
    with conn.cursor() as cur:
        cur.execute(query,params)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()
    return colnames, pnl_data
def fetch_list_alphas(conn, list_name):
    query = """
    SELECT author_key, alpha_key, trong_so 
    FROM danhmuc_alpha 
    WHERE list_name = %s
    """
    
    params = [list_name]
    with conn.cursor() as cur:
        cur.execute(query, params)
        alphas = cur.fetchall()
    
    # Chuyển đổi kết quả thành danh sách các dictionary
    list_alphas = [{'author_key': alpha[0], 'alpha_key': alpha[1], 'trong_so': alpha[2]} for alpha in alphas]
    
    return list_alphas
def fetch_data(conn, list_name):
    query = """
    SELECT d.alpha_key, d.author_key, f.date, d.trong_so, f.total_gain
    FROM danhmuc_alpha d
    JOIN fact_daily f ON d.alpha_key = f.alpha_key
    WHERE d.list_name = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (list_name,))
        result = cur.fetchall()
        columns = ['alpha_key', 'author_key', 'date', 'trong_so', 'total_gain']
        df = pd.DataFrame(result, columns=columns)
        return df

def tinh_toan_thong_so_live(conn, list_name):
    # Tạo DataFrame tổng hợp
    summary_dfs = []
    # Lấy danh sách alpha từ cơ sở dữ liệu
    list_alphas = fetch_list_alphas(conn, list_name)
    for alpha in list_alphas:
        author_key = alpha['author_key']
        alpha_key = alpha['alpha_key']
        trong_so = alpha['trong_so']
        
        # Lấy dữ liệu cho alpha
        data = fetch_data_plot_position_live(conn, author_key, alpha_key)
        df_plot = pd.DataFrame(data, columns=['date', 'position', 'close'])
        df_plot.rename(columns={'date': 'Datetime', 'position': 'Position', 'close': 'Close'}, inplace=True)
        df_plot['Datetime'] = pd.to_datetime(df_plot['Datetime'])
        df_plot.set_index('Datetime', inplace=True)
        
        # Thêm cột phân biệt alpha và trọng số
        df_plot['Alpha'] = alpha_key
        df_plot['trong_so'] = trong_so
        # Tạo đối tượng BacktestInformation và gọi phương thức Plot_PNL
        backtest = BacktestInformation(df_plot.index, df_plot['Position'], df_plot['Close'], fee=0.8)
        summary_df = backtest.Summary01()
        summary_df['alpha_key'] = alpha_key  # Thêm thông tin alpha_key vào summary_df
        summary_dfs.append(summary_df)
    
    # Kết hợp tất cả summary_df vào một DataFrame tổng hợp
    all_summary_df = pd.concat(summary_dfs, ignore_index=True)
    return  all_summary_df

def plot_multiple_total_gain(df):
    df['date'] = pd.to_datetime(df['date'])
    
    # Tính trung bình có trọng số cho mỗi ngày, bỏ qua các ngày không có dữ liệu
    weighted_avg_per_day = df.groupby('date').apply(
        lambda x: (x['total_gain'] * x['trong_so']).sum() / x['trong_so'].sum()
    )
    weighted_avg_df = weighted_avg_per_day.reset_index(name='weighted_avg')
    # Lọc bỏ các ngày có 'weighted_avg' bằng 0
    weighted_avg_df = weighted_avg_df[weighted_avg_df['weighted_avg'] != 0]
    # Vẽ biểu đồ với Plotly
    fig = px.line(weighted_avg_df, x='date', y='weighted_avg', title='PNL giả định')
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=18, family="Arial, sans-serif")
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        width=1500,
        height=550,
        font=dict(family="Arial, sans-serif", size=14),
        template='plotly_dark',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    fig.update_layout(
        shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1, xref='paper', yref='paper', line=dict(color="black", width=1))]
    )
    fig.update_xaxes(tickformat="%Y-%m-%d")
    st.plotly_chart(fig)

        

def fetch_list_alphas_history(conn, list_name):
    query = """
    SELECT author_key, alpha_key, trong_so, add_time, update_time, del_time 
    FROM danhmuc_history 
    WHERE list_name = %s
    """
    
    params = [list_name]
    with conn.cursor() as cur:
        cur.execute(query, params)
        alphas = cur.fetchall()
    
    # Chuyển đổi kết quả thành danh sách các dictionary
    list_alphas = [{'author_key': alpha[0], 'alpha_key': alpha[1], 'trong_so': alpha[2], 'add_time': alpha[3], 'update_time': alpha[4], 'del_time': alpha[5]} for alpha in alphas]
    
    return list_alphas
def plot_multiple_total_gain_history(conn, df_results):
    records = []  # Sử dụng danh sách để lưu trữ các bản ghi

    # Lặp qua từng bản ghi trong df_results
    for index, row in df_results.iterrows():
        alpha_key = row['alpha_key']
        start = row['start']
        end = row['end']
        trong_so = row['trong_so']

        # Lấy dữ liệu từ fact_daily theo điều kiện
        query = """
        SELECT date, total_gain
        FROM fact_daily
        WHERE alpha_key = %s AND date >= %s AND date <= %s
        """
        
        params = [alpha_key, start, end]
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                records_fetched = cur.fetchall()
                
                # Lưu các bản ghi vào danh sách
                for record in records_fetched:
                    records.append({  # Thay vì df_plot.append, ta thêm vào danh sách records
                        'alpha_key': alpha_key,
                        'date': record[0],  # Giả sử cột đầu tiên là date
                        'total_gain': record[1],
                        'trong_so': trong_so
                    })
        except Exception as e:
            print(f"Có lỗi xảy ra trong quá trình truy vấn fact_daily cho alpha_key {alpha_key}: {e}")

    # Chuyển đổi danh sách bản ghi thành DataFrame
    df_plot = pd.DataFrame(records) 
   
    if not df_plot.empty:
        # Tính toán tổng total_gain cho từng ngày
        df_grouped = df_plot.groupby('date').apply(
            lambda x: (x['total_gain'] * x['trong_so']).sum() / x['trong_so'].sum()
        ).reset_index(name='total_gain')
        df_grouped = df_grouped[df_grouped['total_gain'] != 0]
        fig = px.line(df_grouped, x='date', y='total_gain', title='PNL thực tế')
        fig.update_layout(
            title_x=0.5,  # Căn giữa tiêu đề (0: trái, 1: phải, 0.5: giữa)
            title_font=dict(size=18, family="Arial, sans-serif")
        )
        # Cập nhật layout (thêm thanh trượt zoom vào trục x)
        fig.update_layout(
            xaxis_rangeslider_visible=True,
            width=1500,
            height=550,
            font=dict(family="Arial, sans-serif", size=14),
            template='plotly_dark'
        )
        fig.update_layout(
        shapes=[dict(type='rect', x0=0, x1=1, y0=0, y1=1, xref='paper', yref='paper', line=dict(color="black", width=1))])
        
        # Hiển thị biểu đồ trong Streamlit
        # Thêm lưới và căn chỉnh các ngày trên trục x
        fig.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig)
        
################################## HISTORY CHO ALPHA LIVE ########################################



def get_min_max_date_fact_daily(conn, alpha_key):
    query = """
    SELECT MIN(date) AS min_date, MAX(date) AS max_date 
    FROM fact_daily
    WHERE alpha_key = %s
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, [alpha_key])
            result = cur.fetchone()
            min_date = pd.to_datetime(result[0]) if result[0] is not None else pd.NaT
            max_date = pd.to_datetime(result[1]) if result[1] is not None else pd.NaT
            return min_date, max_date
    except Exception as e:
        print(f"Có lỗi xảy ra trong quá trình lấy dữ liệu từ fact_daily: {e}")
        return None, None

def fetch_and_calculate_weights(conn, list_name):
    # Lấy danh sách alpha_key và chi tiết từ lịch sử
    query = """
    SELECT author_key, alpha_key, trong_so, add_time, update_time, del_time 
    FROM danhmuc_history 
    WHERE list_name = %s
    order by id
    """
    
    params = [list_name]
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            alphas = cur.fetchall()
            
        # Chuyển đổi kết quả thành DataFrame
        list_alphas = [
            {
                'author_key': alpha[0],
                'alpha_key': alpha[1],
                'trong_so': alpha[2],
                'add_time': alpha[3],
                'update_time': alpha[4],
                'del_time': alpha[5]
            } for alpha in alphas
        ]
        
        # Nếu không có bản ghi nào, trả về rỗng
        if not list_alphas:
            return []
        
        df = pd.DataFrame(list_alphas)
        # Lọc ra các bản ghi không cần thiết
        unique_alpha_keys = df['alpha_key'].unique()
        results = []
        
        for alpha_key in unique_alpha_keys:
            min_date, max_date = get_min_max_date_fact_daily(conn, alpha_key)
            df_alpha = df[df['alpha_key'] == alpha_key].reset_index(drop=True)
            add_time_to_skip = df_alpha[df_alpha['add_time'] == df_alpha['del_time']]['add_time'].unique()

            df_alpha = df_alpha[~df_alpha['add_time'].isin(add_time_to_skip)].reset_index(drop=True)
            # Tìm các bản ghi có cả update_time và del_time
            valid_records = df_alpha.dropna(subset=['update_time', 'del_time'])

            # Lấy update_time từ các bản ghi hợp lệ
            update_times_to_skip = valid_records['update_time'].unique()
            df_alpha = df_alpha[~df_alpha['update_time'].isin(update_times_to_skip) | (df_alpha['del_time'].notna())].reset_index(drop=True)
            df_alpha = df_alpha.loc[~df_alpha.duplicated(subset=['add_time', 'update_time', 'del_time'], keep='last')].reset_index(drop=True)
            # Tính toán trọng số cho từng khoảng thời gian
            for index, row in df_alpha.iterrows():
                weight = row['trong_so']
                add_time = row['add_time']if not pd.isna(row['add_time']) else min_date
                if add_time < min_date:
                    add_time = min_date
                del_time = row['del_time'] if not pd.isna(row['del_time']) else max_date
                if pd.notna(row['update_time']) :
                    # Trường hợp chỉ có update_time
                    start_time = row['update_time']
                    # Tìm update_time nhỏ nhất lớn hơn start_time
                    next_update_time = df_alpha.loc[df_alpha['update_time'] > start_time, 'update_time'].min()
                   
                    if pd.isna(next_update_time):
                        # Nếu không có update_time nào lớn hơn, tìm del_time lớn hơn start_time
                        next_del_time = del_time
                        end_time = next_del_time 
                       
                    else:

                        end_time = next_update_time - timedelta(days=1)
                    results.append((alpha_key, weight, start_time, end_time))
                elif pd.isna(row['add_time']) and pd.isna(row['update_time']) and pd.isna(row['del_time']):
                    start_time = add_time
                    next_update_time = df_alpha.loc[df_alpha['update_time'] > start_time, 'update_time'].min()
                    next_del_time = df_alpha.loc[df_alpha['del_time'] > start_time, 'del_time'].min()
                    if pd.notna(next_update_time) and pd.notna(next_del_time):
                        # Nếu cả hai đều tồn tại, chọn giá trị có index nhỏ hơn làm `end_time`
                        update_index = df_alpha[df_alpha['update_time'] == next_update_time].index[0]
                        del_index = df_alpha[df_alpha['del_time'] == next_del_time].index[0]
                        if update_index < del_index:
                            end_time = next_update_time - timedelta(days=1)
                        else:
                            end_time = next_del_time 
                    elif pd.notna(next_update_time):
                        end_time = next_update_time - timedelta(days=1)
                    elif pd.notna(next_del_time):
                        end_time = next_del_time 
                    else:
                        end_time = del_time
                    results.append((alpha_key, weight, start_time, end_time))
                elif pd.notna(row['add_time']) and pd.isna(row['update_time']):
                    # Trường hợp có add_time và không có update_time 
                    start_time = row['add_time'] 
                    next_update_time = df_alpha.loc[df_alpha['update_time'] > start_time, 'update_time'].min()
                    next_del_time = df_alpha.loc[df_alpha['del_time'] > start_time, 'del_time'].min()
                    print(alpha_key, next_del_time, next_update_time)
                    if pd.isna(next_update_time) and pd.isna(next_del_time):
                        # Nếu không có update_time nào lớn hơn, tìm del_time lớn hơn start_time
                        end_time = del_time 
                    elif pd.isna(next_update_time) and pd.notna(next_del_time):
                        end_time = next_del_time 
                    elif pd.notna(next_update_time) and pd.isna(next_del_time):
                        end_time = next_update_time  - timedelta(days=1)
                    else:
                        if next_update_time < next_del_time:
                            end_time = next_update_time - timedelta(days=1)
                        else:
                            end_time = next_del_time
                                        
                    results.append((alpha_key, weight, start_time, end_time))
                elif pd.notna(row['update_time']) and pd.notna(row['update_time']):
                    # Trường hợp có cả add_time, update_time và del_time
                    start_time = row['update_time']
                    end_time = row['del_time']
                    results.append((alpha_key, weight, start_time, end_time))
        df_results = pd.DataFrame(results, columns=['alpha_key', 'trong_so', 'start', 'end'])

        return df_results
        
    except Exception as e:
        print(f"Có lỗi xảy ra trong quá trình truy vấn hoặc tính toán: {e}")
### Lọc điều kiện 

# Hàm truy vấn lấy dữ liệu PNL theo điều kiện
def fetch_pnl_by_condition_alpha(conn, **conditions):
    base_query = """
    SELECT d.alpha_key, d.author_key, d.start_date, d.status, d.mark, f.margin, f.mdd_score, f.mdd_percent, 
           f.total_trading_quantity, f.total_profit, f.profit_after_fee, f.trading_quantity_per_day, 
           f.profit_per_day_after_fee, f.return_per_year, f.profit_per_year, f.hitrate, f.hitrate_per_day, 
           f.sharp, f.sharp_after_fee
    FROM fact_pnl01 f
    JOIN dim_alpha d ON f.alpha_key = d.alpha_key
    """
    
    conditions_list = []
    comparisons = {
        "margin": ">",
        "mdd_score": "<",
        "mdd_percent": "<",
        "total_trading_quantity": ">",
        "total_profit": ">",
        "profit_after_fee": ">",
        "trading_quantity_per_day": "<",
        "profit_per_day_after_fee": ">",
        "return_per_year": ">",
        "profit_per_year": ">",
        "hitrate": ">",
        "hitrate_per_day": ">",
        "sharp": ">",
        "sharp_after_fee": ">"
    }
    
    for key, comparison in comparisons.items():
        value = conditions.get(key)
        if value is not None:
            conditions_list.append(f"f.{key} {comparison} {value}")
    
    if conditions_list:
        query = base_query + " WHERE " + " AND ".join(conditions_list)
    else:
        query = base_query

    with conn.cursor() as cur:
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        pnl_data = cur.fetchall()
    
    return colnames, pnl_data

# Hàm hiển thị trường nhập liệu với nhãn và ký tự tùy chỉnh
def display_input_row(symbol, input_label):
    c1, c2 = st.sidebar.columns([1, 13])
    
    with c1:
        st.write("##")
        st.markdown(f"**{symbol}**")
    
    with c2:
        value = st.number_input(input_label, key=input_label, format="%.3f",value = None)
    
    return value


### Quản lý tài khoản 

# Hàm thêm tài khoản
def insert_author (conn, author_key, author_name, password, role):
    query =f"""
            insert into dim_author(author_key, author_name, password, role) values (%s, %s, %s, %s) """
    params = (author_key, author_name, password, role)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()  
# Hàm kiểm tra tồn tại tài khoản
def check_author_exists(conn, user_name):
    query = """
        SELECT EXISTS(SELECT 1 FROM dim_author WHERE author_key = %s);
    """
    params = (user_name,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        exists = cur.fetchone()[0]
    return exists
# Hàm hiển thị tài khoản 
def fetch_dim_author(conn):
    query = """
        SELECT * FROM dim_author;
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        data = cur.fetchall()
    
    return colnames, data

# Hàm xóa tài khoản
def delete_author (conn, user_name):
    query =f"""
            delete from dim_author where author_key = %s """
    params = (user_name,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()  

# Hàm sửa tài khoản
def update_author(conn, data_to_update):
    with conn.cursor() as cursor:
        for author_key, author_name, password, role in data_to_update:
            update_query = """
                UPDATE dim_author
                SET author_name = %s, password = %s, role = %s
                WHERE author_key = %s
            """
            params = (author_name, password, role, author_key)
            cursor.execute(update_query, params)
        # Commit các thay đổi
        conn.commit()


### Quản lý alpha 

# Hàm truy vấn lấy danh sách tác giả từ bảng dim_author
def fetch_authors(conn):
    query = "SELECT author_key FROM dim_author;"
    with conn.cursor() as cur:
        cur.execute(query)
        authors = cur.fetchall()
    return [author[0] for author in authors]




# Hàm thêm alpha

def insert_alpha (conn, alpha_key, author_key, status, mark, start_date, type):
    query =f"""
            insert into dim_alpha(alpha_key, author_key, status, mark, start_date, type) values (%s, %s, %s, %s, %s, %s) """
    params = (alpha_key, author_key,  status, mark, start_date,type)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()  
# Hàm kiểm tra tồn tại alpha
def check_alpha_exists(conn, alpha_key, author_key):
    query = """
        SELECT EXISTS(SELECT 1 FROM dim_alpha WHERE alpha_key = %s and author_key = %s);
    """
    params = (alpha_key, author_key,)
    with conn.cursor() as cur:
        cur.execute(query, params)
        exists = cur.fetchone()[0]
    return exists
# Hàm hiển thị alpha 
def fetch_dim_alpha(conn):
    query = """
        SELECT  al.* 
        FROM dim_alpha al ;
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        data = cur.fetchall()
    
    return colnames, data
def update_data(conn):
    colnames, data = fetch_dim_alpha(conn)
    df = pd.DataFrame(data, columns=colnames)
    return df
# Hàm lấy danh sách danh mục 
def fetch_list_name_from_danhmuc(conn):
    query = """
    SELECT DISTINCT list_name FROM danhmuc_alpha;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        list_name_data = cur.fetchall()
    
    # Chuyển đổi kết quả từ danh sách các tuple thành danh sách các giá trị
    list_name_values = [row[0] for row in list_name_data]
    
    return list_name_values

# Hàm xóa alpha
def delete_alpha (conn, alpha_key, author_key):
    query =f"""
            delete from dim_alpha where alpha_key = %s and author_key = %s """
    params = (alpha_key, author_key)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()  

# Hàm sửa alpha
def update_alpha(conn, alpha_key, author_key, status=None, mark=None, type=None):
    query = """
        UPDATE dim_alpha
        SET status = COALESCE(%s, status),
            mark = COALESCE(%s, mark),
            type = COALESCE(%s,type)
        WHERE alpha_key = %s AND author_key = %s
    """
    params = (status, mark,type, alpha_key, author_key)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()
    
    return cur.rowcount > 0
def update_alpha_bulk(conn, data_to_update):
    with conn.cursor() as cursor:
        for alpha_key, author_key, status, mark, start_date, type in data_to_update:
            if mark != None:
                # Lấy danh sách mark hiện tại từ bảng dim_alpha
                cursor.execute("""
                    SELECT mark FROM dim_alpha WHERE alpha_key = %s AND author_key = %s
                """, (alpha_key, author_key))
                existing_marks_row = cursor.fetchone()
                if existing_marks_row:
                    # Kiểm tra giá trị của existing_marks_row
                    existing_marks_value = existing_marks_row[0]
                    # Tách danh sách mark hiện tại
                    existing_marks = set(existing_marks_value.split(',')) if existing_marks_value else set()
                else:
                    existing_marks = set()
                # Tách danh sách mark mới
                new_marks = set(mark.split(','))
                # Xác định các mark mới cần thêm
                marks_to_add = new_marks - existing_marks
                for new_mark in marks_to_add:
                    # Kiểm tra xem mark mới có trùng với list_name trong danhmuc_alpha không
                    cursor.execute("""
                        SELECT list_author
                        FROM danhmuc_alpha
                        WHERE list_name = %s
                    """, (new_mark,))
                    list_author_row = cursor.fetchone()
                    if list_author_row:
                        list_author = list_author_row[0]

                        # Thêm dòng mới vào bảng `danhmuc_alpha` nếu mark mới trùng với list_name
                        insert_query = """
                            INSERT INTO danhmuc_alpha (alpha_key, author_key, list_name, list_author, trong_so)
                            VALUES (%s, %s, %s, %s, %s)
                        """
                        insert_params = (alpha_key, author_key, new_mark, list_author, 1)
                        cursor.execute(insert_query, insert_params)
            # Cập nhật dữ liệu trong bảng `dim_alpha`
            update_query = """
                UPDATE dim_alpha
                SET status = %s, mark = %s, start_date = %s, type = %s
                WHERE alpha_key = %s AND author_key = %s
                RETURNING alpha_key, author_key, mark
            """
            params = (status, mark, start_date, type, alpha_key, author_key)
            cursor.execute(update_query, params)
        # Commit các thay đổi
        conn.commit()

## QUẢN LÝ DANH MỤC
#Thêm danh mục mới
def insert_new_danhmuc(conn, list_name, list_author, alpha_key, author_key, trong_so, list_date, add_time):
    query = """
        INSERT INTO danhmuc_alpha (list_name, list_author, alpha_key, author_key, trong_so, list_date, add_time) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = (list_name, list_author, alpha_key, author_key, trong_so, list_date, add_time)
    with conn.cursor() as cur:
        cur.execute(query, params)
        conn.commit()
# Xóa danh mục
def del_danhmuc(conn, list_name, list_author):
        
        query_del = """
            DELETE FROM danhmuc_alpha
            WHERE list_name = %s AND list_author = %s
        """
        params_del = (list_name, list_author)
        with conn.cursor() as cur:
            cur.execute(query_del, params_del)
            conn.commit()

def remove_mark_for_alpha(conn, alpha_key, remove_mark):
    ''' Xóa các danh mục khỏi cột mark cho alpha_key '''
    with conn.cursor() as cursor:
        # Lấy danh sách mark hiện tại
        select_query = """
            SELECT mark
            FROM dim_alpha
            WHERE alpha_key = %s
        """
        cursor.execute(select_query, (alpha_key,))
        result = cursor.fetchone()
        if result:
            current_marks = result[0]
            current_mark_list = current_marks.split(',') if current_marks else []
            
            # Tìm danh mục cần xóa và loại bỏ khỏi danh sách hiện tại
            remove_mark_list = [mark.strip() for mark in remove_mark.split(',') if mark.strip()]
            
            updated_mark_list = [mark for mark in current_mark_list if mark.strip() not in [r_mark.strip() for r_mark in remove_mark_list]]

            updated_marks = ', '.join(updated_mark_list)
            # Cập nhật lại danh sách mark mới sau khi đã xóa
            update_query = """
                UPDATE dim_alpha
                SET mark = %s
                WHERE alpha_key = %s
            """
            cursor.execute(update_query, (updated_marks, alpha_key))
            conn.commit()


def get_alpha_keys_for_list(conn, list_name, list_author):
    ''' Lấy danh sách alpha_key liên quan đến list_name và list_author '''
    query = """
        SELECT DISTINCT alpha_key
        FROM danhmuc_alpha
        WHERE list_name = %s AND list_author = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (list_name, list_author))
        alpha_keys = cur.fetchall()
    return [alpha_key[0] for alpha_key in alpha_keys]


def update_mark_after_delete(conn, alpha_keys, list_name):
    ''' Cập nhật cột mark trong bảng dim_alpha sau khi xóa list_name '''
    for alpha_key in alpha_keys:
        # Cập nhật cột mark cho từng alpha_key
        remove_mark_for_alpha(conn, alpha_key, list_name)       


def del_danhmuc_history(conn, list_name, list_author):
    query = """
                DELETE FROM danhmuc_history
                WHERE list_name = %s AND list_author = %s
            """
    params = (list_name, list_author)
    with conn.cursor() as cur:
            cur.execute(query, params)
            conn.commit()
# Xử lý khi xóa 1 alpha
def get_alpha_history(conn, list_name, alpha_key, author_key):
    query = """
    SELECT * FROM danhmuc_history
    WHERE list_name = %s AND alpha_key = %s AND author_key = %s
    ORDER BY add_time ASC
    """
    params = (list_name, alpha_key, author_key)
    df_history = pd.read_sql(query, conn, params=params)
    return df_history
def update_danhmuc_history1(conn, alpha_updates):
    query = """
    INSERT INTO danhmuc_history (
        list_name, list_author, alpha_key, author_key, trong_so, list_date, add_time, del_time, update_time
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id)
    DO UPDATE SET
        del_time = EXCLUDED.del_time,
        update_time = EXCLUDED.update_time
    """
    
    def parse_date(date_value):
        if isinstance(date_value, str):
            date_value = date_value.split('T')[0]  # Chỉ lấy phần ngày nếu có thời gian
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        return date_value
    
    # Xử lý dữ liệu
    params_list = []
    for update in alpha_updates:
        del_time = parse_date(update.get('del_time', None))
        update_time = parse_date(update.get('update_time', None))
        
        params = (
            update.get('list_name'),
            update.get('list_author'),
            update.get('alpha_key'),
            update.get('author_key'),
            int(update.get('trong_so', 0)),  # Chuyển đổi trọng số thành int
            update.get('list_date'),
            update.get('add_time'),
            del_time,
            update_time
        )
        params_list.append(params)
    
    # Thực hiện cập nhật
    with conn.cursor() as cursor:
        cursor.executemany(query, params_list)
        conn.commit()






def update_mark_for_alpha(conn, alpha_key, new_mark):
    ''' Cập nhật mark cho alpha_key với danh mục mới '''
    with conn.cursor() as cursor:
        # Lấy danh sách mark hiện tại
        select_query = """
            SELECT mark
            FROM dim_alpha
            WHERE alpha_key = %s
        """
        cursor.execute(select_query, (alpha_key,))
        result = cursor.fetchone()

        if result:
            current_marks = result[0]
            current_mark_list = current_marks.split(',') if current_marks else []

            # Tìm danh mục mới không có trong danh sách hiện tại
            new_mark_list = [mark.strip() for mark in new_mark.split(',') if mark.strip()]
            updated_mark_list = list(set(current_mark_list + new_mark_list))
            updated_marks = ', '.join(updated_mark_list)

            # Cập nhật danh sách mark mới
            update_query = """
                UPDATE dim_alpha
                SET mark = %s
                WHERE alpha_key = %s
            """
            cursor.execute(update_query, (updated_marks, alpha_key))
            conn.commit()


# Hàm xóa alpha
def delete_danhmuc_alpha(conn, list_name, alpha_key, author_key):
    # Xóa dữ liệu từ bảng danhmuc_alpha
    delete_query = """
        DELETE FROM danhmuc_alpha
        WHERE list_name = %s and alpha_key = %s AND author_key = %s  
    """
    params = (list_name, alpha_key, author_key)
    with conn.cursor() as cur:
        cur.execute(delete_query, params)
        conn.commit()
    
    # Cập nhật bảng dim_alpha
    update_query = """
        UPDATE dim_alpha
        SET mark = (
            SELECT STRING_AGG(list_name, ',') 
            FROM danhmuc_alpha 
            WHERE alpha_key = dim_alpha.alpha_key AND author_key = dim_alpha.author_key
        )
        WHERE alpha_key = %s AND author_key = %s
    """
    params_update = (alpha_key, author_key)
    with conn.cursor() as cur:
        cur.execute(update_query, params_update)
        conn.commit()


def update_trong_so_bulk1(conn, data):
    st.write(data)
    query = """
                UPDATE danhmuc_alpha
                SET trong_so = %s
                WHERE alpha_key = %s AND author_key = %s AND list_name = %s;
            """

    with conn.cursor() as cur:
        cur.executemany(query, data)  # Thực hiện nhiều lệnh cập nhật cùng lúc
        conn.commit()
def fetch_list_author(conn):
    query = """select distinct list_author from danhmuc_alpha """
    with conn.cursor() as cur:
        cur.execute(query)
        authors = cur.fetchall()
    return [author[0] for author in authors]


def fetch_list_name(conn):
    query = f"""select distinct list_name from danhmuc_alpha """
    with conn.cursor() as cur:
        cur.execute(query)
        names = cur.fetchall()
    return [name[0] for name in names]


# Hàm hiển thị dữ liệu theo dạng dọc
def display_data_vertically(colnames, pnl_data):
    # Tạo DataFrame từ dữ liệu lấy được
    df = pd.DataFrame(pnl_data, columns=colnames)
    return df

def hien_thi_bang_du_lieu(df):
    df['trong_so'] = 1.0
    
    grid_options = GridOptionsBuilder.from_dataframe(df)
    grid_options.configure_grid_options(rowSelection='multiple')
    grid_options.configure_selection('multiple', use_checkbox=True)
    grid_options.configure_default_column(editable=True, filter=True, sortable=True)
    grid_options.configure_column('trong_so', editable=True)
    grid_options.configure_side_bar()
    grid_options = grid_options.build()

    grid_response = AgGrid(df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                editable=True,
                update_mode = GridUpdateMode.MANUAL,  
                fit_columns_on_grid_load=True,
                height=600
            )

    return grid_response

# Hàm hiển thị alpha 
def fetch_alpha_fact_pnl(conn):
    query = """
        SELECT  d.*  
        FROM fact_pnl01 f join dim_alpha d on f.alpha_key = d.alpha_key;
    """
    
    with conn.cursor() as cur:
        cur.execute(query)
        colnames = [desc[0] for desc in cur.description]
        data = cur.fetchall()
    
    return colnames, data
def hien_thi_bang_du_lieu_2(df):
    if df is None:
        return
    else:
        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_selection('single')
        grid_options.configure_side_bar()
        grid_options = grid_options.build()

        grid_response = AgGrid(df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    editable=True,
                    update_mode=GridUpdateMode.MODEL_CHANGED, 
                    fit_columns_on_grid_load=True,
                    height=200
                )

        return grid_response
def hien_thi_bang_du_lieu_3(df):
    grid_options = GridOptionsBuilder.from_dataframe(df)
    grid_options.configure_selection('single')
    grid_options.configure_column('trong_so', editable=True)
    grid_options.configure_side_bar()
    grid_options = grid_options.build()

    grid_response = AgGrid(df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                editable=True,
                update_mode=GridUpdateMode.MANUAL, 
                fit_columns_on_grid_load=True,
                height=200
            )

    return grid_response
# Hàm chính của ứng dụng Streamlit
def main():
    
    function_choice = None
    function_choice_user = None
    # Kiểm tra trạng thái đăng nhập
    if "author_key" not in st.session_state:
        # Hiển thị form đăng nhập nếu chưa đăng nhập
        cols = st.columns(3)

        # the 3rd column
        with cols[1]:
            st.markdown(
                    """
                    <style>
                    .custom-title {
                        text-align: center;
                    }
                    </style>
                    <h1 class="custom-title">Đăng nhập</h1>
                    """,
                    unsafe_allow_html=True
                )

            author_key = st.text_input("Author_key")
            password = st.text_input("Password", type="password")
            cols = st.columns(3)
            with cols[1]:
                if st.button("Đăng nhập"):
                    user_data = authenticate(author_key, password)  # Giả sử có hàm authenticate
                    if user_data:
                        st.session_state["author_key"] = user_data["author_key"]
                        st.session_state["role"] = user_data["role"]
                        st.success(f"Chào mừng {author_key}!")
                        st.experimental_rerun()
                    else:
                        st.error("Tài khoản hoặc mật khẩu sai")
    else:
        # Hiển thị tiêu đề và nút logout
        st.markdown(
            """
            <style>
            /* Căn giữa tiêu đề */
            .custom-title {
                text-align: center;
            }
            </style>
            <h1 class="custom-title">Quản lý Alpha và PNL</h1>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.write(f"Tài khoản đăng nhập hiện tại: {st.session_state['author_key']}")

        if st.sidebar.button("Đăng xuất", key="logout_button"):
            del st.session_state["author_key"]
            del st.session_state["role"]
            st.experimental_rerun()
            # Hiển thị sidebar và các chức năng khi đã đăng nhập
    
        
        user_role = st.session_state["role"]
        author_key = st.session_state["author_key"] 

        if user_role == "admin":
            function_choice = st.sidebar.selectbox("Chọn chức năng", [
                "Xem position",
                "Xem thông số", 
                "Lọc điều kiện",  
                "Quản lý tài khoản",
                "Quản lý alpha",
                "Quản lý danh mục"
            ])
        elif user_role == "user":
            function_choice_user = st.sidebar.selectbox("Chọn chức năng", [
                "Xem position",
                "Xem thông số"
            ])
        
        # Kết nối tới cơ sở dữ liệu (Giả sử có hàm connect_to_db)
        conn = connect_to_db()
##########################################################
############## USER ###################################




# Xem position
    if function_choice_user == "Xem position":
        selected_status = st.sidebar.selectbox("Chọn trạng thái", ["Xem tất cả", "live", "paper", "đã tắt", "wait list"])
        status_key = selected_status if selected_status != "Xem tất cả" else None
        colnames, pnl_data = fetch_position_by_author(conn, author_key,status_key)
        st.write(f"Dữ liệu Position cho Tác giả: {author_key}")
        
        df = pd.DataFrame(pnl_data, columns=colnames)
        df = df[~df['alpha_key'].str.startswith('Intraday_')]
        grid_options = GridOptionsBuilder.from_dataframe(df)
        grid_options.configure_selection('single')
        grid_options.configure_side_bar()
        grid_options.configure_default_column(editable=True)
        grid_options = grid_options.build()

        grid_response = AgGrid(df, gridOptions=grid_options,fit_columns_on_grid_load=True, enable_enterprise_modules=True)
        st.write("Tổng của cột 'position':", df['position'].sum())

#Xem thông số
    
    elif function_choice_user == "Xem thông số":
        # Hiển thị các nút hành động     
        selected_status = st.sidebar.selectbox("Chọn trạng thái", ["Xem tất cả", "live", "paper", "đã tắt", "wait list"])
        status_key = selected_status if selected_status != "Xem tất cả" else None
        colnames, pnl_data = fetch_pnl_by_author(conn, author_key,status_key)
        st.write(f"Dữ liệu PNL cho Tác giả: {author_key}")
        df = pd.DataFrame(pnl_data, columns=colnames)
        df = df[~df['alpha_key'].str.startswith('Intraday_')]
       
        # Tạo cấu hình cho AgGrid
        grid_options = GridOptionsBuilder.from_dataframe(df)
        for col in df.columns:
            grid_options.configure_column(col, cellStyle=cellstyle_jscode)
        grid_options.configure_selection('single')
        grid_options.configure_side_bar()
        grid_options.configure_default_column(editable=True, filter=True, sortable=True)
        grid_options = grid_options.build()
        
        # Hiển thị bảng với các ô được làm nổi bật
        grid_response = AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True, enable_enterprise_modules=True, allow_unsafe_jscode=True)
        st.write("Tổng của cột 'position':", df['position'].sum())
        selected_row = grid_response['selected_rows']
    
        if not isinstance(selected_row, pd.DataFrame) :
            
            mean = pnl_mean(conn, author_key, status_key)
            if mean is not None:
                df_all = fetch_data_all (conn, author_key, status_key)
                if df_all is not None: 
                    plot_all_total_gain(df_all)
                    hien_thi_bang_du_lieu_2(mean)
                else:
                    st.write('Không có alpha nào')

            else:
                st.warning("Không có dữ liệu alpha để hiển thị.")

        else:
            plt.close('all')  
            # Kiểm tra kiểu dữ liệu của selected_rows
            if isinstance(selected_row, pd.DataFrame) and len(selected_row) > 0:
                selected_row = pd.DataFrame(selected_row)
                
                # Kiểm tra nếu DataFrame không rỗng và có ít nhất một hàng
                if not selected_row.empty and selected_row.shape[0] > 0:
                    # Truy cập cột alpha_key
                    if 'alpha_key' in selected_row.columns:
                        selected_row_data = selected_row['alpha_key']
                        alpha_key = selected_row_data.iloc[0]
                        st.write(f"Dữ liệu PNL cho Alpha: {alpha_key}")

                        if alpha_key:
                            if check_alpha_in_fact_positions(conn, alpha_key):
                                    # Nếu alpha_key có trong fact_positions01
                                    data = fetch_data_plot_position_live(conn, author_key, alpha_key) 
                                    df_plot = pd.DataFrame(data, columns=['date', 'position', 'close'])
                                    df_plot.rename(columns={'date': 'Datetime', 'position': 'Position', 'close': 'Close'}, inplace=True)
                                    df_plot['Datetime'] = pd.to_datetime(df_plot['Datetime'])
                                    backtest = BacktestInformation(df_plot['Datetime'], df_plot['Position'], df_plot['Close'], fee=0.8)
                                    backtest.Plot_PNL()
                                    summary_df = backtest.Summary01()                 
                                    summary_df['author_key'] = author_key
                                    summary_df['alpha_key'] = alpha_key
                                    combined_df = pd.concat([summary_df], axis=1)                
                                    st.write(combined_df)
                            else:
                                # Nếu alpha_key không có trong fact_positions01
                                colnames, pnl_data = fetch_pnl_by_alpha(conn, alpha_key)
                                plot_total_gain(conn, author_key, alpha_key)
                                display_df = display_data_vertically(colnames, pnl_data)
                                st.write(display_df)
                        else:
                            st.error("Không tìm thấy thông tin 'Alpha' trong hàng được chọn.")
                    else:
                        st.error("Cột 'alpha_key' không tồn tại trong dữ liệu được chọn.")
                else:
                    st.error("Không có hàng nào được chọn.")
            else:
                st.error("Không có hàng nào được chọn.")
        

####################### XEM POSITIONS #####################
    if function_choice == "Xem position":
        # Hiển thị các nút hành động
        st.sidebar.header("Lựa chọn chức năng")
        thongso_action = st.sidebar.selectbox("Chọn chức năng", ["Xem theo tác giả", "Xem theo danh mục"])
        if thongso_action == 'Xem theo tác giả':
            # Hiển thị các nút hành động
            st.sidebar.header("Lựa chọn position")
            authors = fetch_authors_by_fact_positions01(conn)
            authors.insert(0, "Xem tất cả tác giả")
            selected_author = st.sidebar.selectbox("Chọn tác giả", authors)
            selected_status = st.sidebar.selectbox("Chọn trạng thái", ["Xem tất cả", "live", "paper", "đã tắt", "wait list", "None"])
            status_key = selected_status if selected_status != "Xem tất cả" else None
            author_key = selected_author if selected_author != "Xem tất cả tác giả" else None
            colnames, pnl_data = fetch_position_by_author(conn, author_key,status_key)
            st.write(f"Dữ liệu Position cho Tác giả: {selected_author}")
            
            df = pd.DataFrame(pnl_data, columns=colnames)
            df = df[~df['alpha_key'].str.startswith('Intraday_')]
            grid_options = GridOptionsBuilder.from_dataframe(df)
            grid_options.configure_selection('single')
            grid_options.configure_side_bar()
            grid_options.configure_default_column(editable=True)
            grid_options = grid_options.build()

            grid_response = AgGrid(df, gridOptions=grid_options,fit_columns_on_grid_load=True, enable_enterprise_modules=True, update_mode=GridUpdateMode.COLUMN_CHANGED)
            st.write("Tổng của cột 'position':", df['position'].sum())
        else:
            hien_thi_position_danh_muc(conn)


############################## XEM THÔNG SỐ ###########################################
    
    elif function_choice == "Xem thông số":
        # Hiển thị các nút hành động
        st.sidebar.header("Lựa chọn chức năng")
        thongso_action = st.sidebar.selectbox("Chọn chức năng", ["Xem theo PNL", "Xem theo danh mục"])
        if thongso_action == 'Xem theo PNL':
            authors = fetch_authors_by_fact_pnl(conn)
            authors.insert(0, "Xem tất cả tác giả")
            selected_author = st.sidebar.selectbox("Chọn tác giả", authors)
            selected_status = st.sidebar.selectbox("Chọn trạng thái", ["Xem tất cả", "live", "paper", "đã tắt", "wait list", " "])
            status_key = selected_status if selected_status != "Xem tất cả" else None
            author_key = selected_author if selected_author != "Xem tất cả tác giả" else None
            colnames, pnl_data = fetch_pnl_by_author(conn, author_key, status_key)
            st.write(f"Dữ liệu PNL cho Tác giả: {selected_author if selected_author else 'Tất cả tác giả'}")

            df = pd.DataFrame(pnl_data, columns=colnames)
            df = df[~df['alpha_key'].str.startswith('Intraday_')]
            # Áp dụng màu sắc cho bảng dữ liệu

            
            # Tạo cấu hình cho AgGrid
            grid_options = GridOptionsBuilder.from_dataframe(df)
            for col in df.columns:
                grid_options.configure_column(col, cellStyle=cellstyle_jscode)
            grid_options.configure_selection('single')
            grid_options.configure_side_bar()
            grid_options.configure_default_column(editable=True, filter=True, sortable=True)
            grid_options = grid_options.build()
            
            # Hiển thị bảng với các ô được làm nổi bật
            grid_response = AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True, enable_enterprise_modules=True, allow_unsafe_jscode=True)
            st.write("Tổng của cột 'position':", df['position'].sum())
            selected_row = grid_response['selected_rows']
           
            if not isinstance(selected_row, pd.DataFrame) :
                mean = pnl_mean(conn, author_key, status_key)
                if not mean.isna().all().all():
                   
                    df_all = fetch_data_all (conn, author_key, status_key)
                    if df_all is not None: 
                        plot_all_total_gain(df_all)
                        hien_thi_bang_du_lieu_2(mean)
                    else:
                        st.write('Không có alpha nào')
            else:
                plt.close('all')  
                # Kiểm tra kiểu dữ liệu của selected_rows
                if  len(selected_row) > 0:
                    selected_row = pd.DataFrame(selected_row)
                    
                    # Kiểm tra nếu DataFrame không rỗng và có ít nhất một hàng
                    if  selected_row.shape[0] > 0:
                        # Truy cập cột alpha_key
                        if 'alpha_key' in selected_row.columns:
                            selected_row_data = selected_row['alpha_key']
                            alpha_key = selected_row_data.iloc[0]
                            selected_row_data = selected_row['author_key']
                            author_key = selected_row_data.iloc[0]

                            st.write(f"Dữ liệu PNL cho Alpha: {alpha_key}")

                            if alpha_key:
                                if check_alpha_in_fact_positions(conn, alpha_key):
                                    # Nếu alpha_key có trong fact_positions01
                                    data = fetch_data_plot_position_live(conn, author_key, alpha_key) 
                                    df_plot = pd.DataFrame(data, columns=['date', 'position', 'close'])
                                    df_plot.rename(columns={'date': 'Datetime', 'position': 'Position', 'close': 'Close'}, inplace=True)
                                    df_plot['Datetime'] = pd.to_datetime(df_plot['Datetime'])
                                    backtest = BacktestInformation(df_plot['Datetime'], df_plot['Position'], df_plot['Close'], fee=0.8)
                                    plot_total_gain(conn,author_key,alpha_key)
                                    summary_df = backtest.Summary01() 
                                    
                                    summary_df['author_key'] = author_key
                                    summary_df['alpha_key'] = alpha_key
                                    # Kết hợp các DataFrame để tạo combined_df
                                    combined_df = pd.concat([summary_df], axis=1)                
                                    # Hiển thị DataFrame kết hợp bằng Streamlit
                                    st.write(combined_df)
                                else:
                                    # Nếu alpha_key không có trong fact_positions01
                                    colnames, pnl_data = fetch_pnl_by_alpha(conn, alpha_key)
                                    plot_total_gain(conn, author_key, alpha_key)
                                    display_df = display_data_vertically(colnames, pnl_data)
                                    grid_response1 = hien_thi_bang_du_lieu_2(display_df)
                            else:
                                st.error("Không tìm thấy thông tin 'Alpha' trong hàng được chọn.")
                        else:
                            st.error("Cột 'alpha_key' không tồn tại trong dữ liệu được chọn.")
                    else:
                        st.error("Không có hàng nào được chọn.")
                else:
                    st.error("Không có hàng nào được chọn.")
        else: 
            df_thong_so_danh_muc = hien_thi_thong_so_danh_muc(conn)
            # Tạo grid options cho AgGrid
            grid_options = GridOptionsBuilder.from_dataframe(df_thong_so_danh_muc)
            grid_options.configure_selection('single')  
            grid_options.configure_side_bar()  
            grid_options.configure_default_column(editable=True, filter=True, sortable=True)  
            grid_options = grid_options.build()
            
            # Hiển thị AgGrid
            
            grid_response = AgGrid(df_thong_so_danh_muc, gridOptions=grid_options, fit_columns_on_grid_load=True, enable_enterprise_modules=True)
            selected_row = grid_response['selected_rows']
            if isinstance(selected_row, pd.DataFrame) and len(selected_row) > 0:
                    selected_row = pd.DataFrame(selected_row)
                    
                    # Kiểm tra nếu DataFrame không rỗng và có ít nhất một hàng
                    if not selected_row.empty and selected_row.shape[0] > 0:
                        # Truy cập cột alpha_key
                        if 'list_name' in selected_row.columns:
                            selected_row_data = selected_row['list_name']
                            list_name = selected_row_data.iloc[0]
                            st.write(f"Dữ liệu PNL cho danh mục: {list_name}")

                            if list_name:
                                list_alphas = fetch_list_alphas(conn, list_name)
                                
                                all_alphas_exist = all(check_alpha_in_fact_positions(conn, alpha['alpha_key']) for alpha in list_alphas)
                                

                                # Tính toán và in ra thời gian đã trôi qua
                                
                                if all_alphas_exist:
                                    colnames, pnl_data = fetch_pnl_list_name_live(conn, list_name)
                                    display_df = display_data_vertically(colnames, pnl_data)
                                    # Gọi hàm plot_multiple_pnls và nhận kết quả
                                    start_time = time.time()
                                    summary_df = tinh_toan_thong_so_live(conn, list_name)
                                    end_time = time.time()
                                    elapsed_time = end_time - start_time
                                    st.write(f"Time taken: {elapsed_time:.4f} seconds")
                                    data_daily = fetch_data(conn, list_name)
                                    plot_multiple_total_gain(data_daily)

                                    # Đảm bảo rằng display_df có cột alpha_key nếu cần thiết
                                    if  'alpha_key' in summary_df.columns:
                                        new_column_order = ['author_key', 'alpha_key','position','trong_so', 'profit', 'profit_M', 'profit_3M', 'margin', 'margin_M', 'margin_3M', 
                                                            'mdd_score', 'mdd_percent', 'trad_per_day', 'sharp', 'sharp_M', 'sharp_3M', 
                                                            'hit_per_day', 'hitrate', 'hit_M', 'hit_3M', 'return', 'return_M', 'return_3M']
                                        combined_df = pd.merge(display_df, summary_df, on='alpha_key', how='outer')
                                        combined_df = combined_df.reindex(columns=new_column_order)
                                        st.write(combined_df)
                                        st.write("Tổng của cột 'position':", combined_df['position'].sum())
                                    else:
                                        st.write("Không có dữ liệu alpha để hiển thị.")
                                    
                                else:
                                    colnames, pnl_data = fetch_pnl_list_name(conn, list_name)
                                    display_df = display_data_vertically(colnames, pnl_data)
                                    data_daily = fetch_data(conn, list_name)
                                    plot_multiple_total_gain(conn,  data_daily)
                                    grid_response1 = hien_thi_bang_du_lieu_3(display_df)
                                
                            else:
                                st.error("Không tìm thấy thông tin 'list_name' trong hàng được chọn.")
                        else:
                            st.error("Cột 'list_name' không tồn tại trong dữ liệu được chọn.")
                    else:
                        st.error("Không có hàng nào được chọn.")
            else:
                st.error("Không có hàng nào được chọn.")

############################### LỌC ĐIỀU KIỆN #################################################
    elif function_choice == "Lọc điều kiện":
        st.sidebar.header("Lọc Alpha theo điều kiện") 
        # Các trường nhập liệu
        fields = {
            "margin": (">", "Nhập giá trị margin"),
            "mdd_score": ("<", "Nhập giá trị mdd_score"),
            "mdd_percent": ("<", "Nhập giá trị mdd_percent"),
            "total_trading_quantity": (">", "Nhập giá trị totaL_trading_quantity"),
            "total_profit": (">", "Nhập giá trị total_profit"),
            "profit_after_fee": (">", "Nhập giá trị profit_after_fee"),
            "trading_quantity_per_day": ("<", "Nhập giá trị trading_quantity_per_day"),
            "profit_per_day_after_fee": (">", "Nhập giá trị profit_per_day_after_fee"),
            "return_per_year": (">", "Nhập giá trị return_per_year"),
            "profit_per_year": (">", "Nhập giá trị profit_per_year"),
            "hitrate": (">", "Nhập giá trị hitrate"),
            "hitrate_per_day": (">", "Nhập giá trị hitrate_per_day"),
            "sharp": (">", "Nhập giá trị sharp"),
            "sharp_after_fee": (">", "Nhập giá trị sharp_after_fee")
        }      
        inputs = {key: display_input_row(symbol, label) for key, (symbol, label) in fields.items()}
        if st.sidebar.button("Xem kết quả"):
            # Gọi hàm fetch_pnl_by_condition_alpha và xử lý dữ liệu
            colnames, pnl_data = fetch_pnl_by_condition_alpha(conn, **inputs)
            st.write("Dữ liệu PNL sau khi lọc theo điều kiện")
            df = pd.DataFrame(pnl_data, columns=colnames)
            df = df[~df['alpha_key'].str.startswith('Intraday_')]
            st.data_editor(df, use_container_width=True)


######################## QUẢN LÝ TÀI KHOẢN #########################################
    elif function_choice == "Quản lý tài khoản":
        st.sidebar.header("Quản lý tài khoản")
        author_action = st.sidebar.radio("Chọn hành động", ["Xem và chỉnh sửa", "Thêm tài khoản mới"])
        if author_action == "Xem và chỉnh sửa":
                colnames, data = fetch_dim_author(conn)
                st.write("Danh sách tài khoản:")
                df = pd.DataFrame(data, columns=colnames)

                grid_options = GridOptionsBuilder.from_dataframe(df)
                grid_options.configure_selection('single')
                grid_options.configure_side_bar()

                grid_options.configure_column('role', editable=True, cellEditor='agSelectCellEditor', cellEditorParams={'values': ['admin','user']})
                grid_options.configure_column('author_name', editable=True, cellEditor='agTextCellEditor')
                grid_options.configure_column('password', editable=True, cellEditor='agTextCellEditor')

                grid_options = grid_options.build()
                grid_response = AgGrid(df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    editable=True,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    fit_columns_on_grid_load=True,
                    height=600
                )

                if st.button("Lưu dữ liệu"):
                    # Lấy dữ liệu đã chỉnh sửa từ bảng
                    edited_df = pd.DataFrame(grid_response['data'])
                     # Tìm các dòng đã thay đổi
                    merged_df = pd.merge(df, edited_df, how='outer', indicator=True)
                    changed_rows = merged_df[merged_df['_merge'] == 'right_only']
                    # Chuyển dữ liệu thành danh sách để cập nhật
                    data_to_update = [
                        (row.get('author_key', None), row.get('author_name', None), row.get('password', None), row.get('role', None))
                        for index, row in changed_rows.iterrows()
                    ]
                    # Thực hiện cập nhật dữ liệu
                    update_author(conn, data_to_update)

                    # Xác nhận cập nhật thành công
                    st.success("Dữ liệu đã được lưu thành công!")  # Thông báo khi lưu thành công
              

                selected_row = grid_response['selected_rows']

                if len(selected_row) > 0:
                    selected_row_df = pd.DataFrame(selected_row)
                    author_key = selected_row_df['author_key'].iloc[0]
                if st.button("Xóa"):
                    delete_author(conn, author_key)
                    st.experimental_rerun()  # Reload lại trang để cập nhật bảng
                
            
        elif author_action == "Thêm tài khoản mới":
            st.sidebar.subheader("Thêm tài khoản mới")
            new_authorkey = st.sidebar.text_input("Nhập key của tác giả")
            new_authorname = st.sidebar.text_input("Tên tác giả")
            new_password = st.sidebar.text_input("Mật khẩu", type="password")
            new_role = st.sidebar.selectbox("Chọn quyền", [ "admin","user"])
            if st.sidebar.button("Thêm"):
                if new_authorkey and new_authorname and new_password and new_role:
                    if check_author_exists(conn, new_authorkey):
                        st.error("Tên tài khoản đã tồn tại.")
                    else:
                        insert_author(conn, new_authorkey, new_authorname, new_password, new_role)
                        st.success(f"Đã thêm tác giả {new_authorkey}")
                    colnames, pnl_data = fetch_dim_author(conn)
                    df = pd.DataFrame(pnl_data, columns=colnames)
                   
                    st.data_editor(df, use_container_width=True)
                
                else:
                    st.error("Vui lòng điền đầy đủ thông tin.")


######################################## QUẢN LÝ ALPHA ##################################################
    elif function_choice == "Quản lý alpha":
        st.sidebar.header("Quản lý alpha")
        alpha_action = st.sidebar.radio("Chọn hành động", ["Xem và chỉnh sửa", "Thêm alpha"])
        if alpha_action == "Xem và chỉnh sửa":
                colnames, data = fetch_dim_alpha(conn)
                st.write("Danh sách alpha:")
                df = pd.DataFrame(data, columns=colnames)
                df = df[~df['alpha_key'].str.startswith('Intraday_')]
                grid_options = GridOptionsBuilder.from_dataframe(df)
                grid_options.configure_selection('single')
                grid_options.configure_side_bar()

                grid_options.configure_column('status', editable=True, cellEditor='agSelectCellEditor', cellEditorParams={'values': ['live', 'paper', 'đã tắt', 'wait list', ' ']})
                list_name_values = fetch_list_name_from_danhmuc(conn)
                grid_options.configure_column('mark', editable=True, cellEditor='agTextCellEditor', cellEditorParams={'values': list_name_values})
                grid_options.configure_column('start_date', editable=True, cellEditor='agTextCellEditor')
                grid_options.configure_column('type', editable=True, cellEditor='agSelectCellEditor', cellEditorParams={'values': ['TA', 'ML', 'kết hợp']})

                grid_options = grid_options.build()
                grid_response = AgGrid(df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    editable=True,
                    update_mode=GridUpdateMode.MANUAL,
                    fit_columns_on_grid_load=True,
                    height=600
                )

                if st.button("Lưu dữ liệu"):
                    # Lấy dữ liệu đã chỉnh sửa từ bảng
                    edited_df = pd.DataFrame(grid_response['data'])
                     # Tìm các dòng đã thay đổi
                    merged_df = pd.merge(df, edited_df, how='outer', indicator=True)
                    changed_rows = merged_df[merged_df['_merge'] == 'right_only']
                    # Chuyển dữ liệu thành danh sách để cập nhật
                    data_to_update = [
                        (row.get('alpha_key', None), row.get('author_key', None), row.get('status', None), row.get('mark', None), row.get('start_date', None), row.get('type', None))
                        for index, row in changed_rows.iterrows()
                    ]

                    # Thực hiện cập nhật dữ liệu
                    update_alpha_bulk(conn, data_to_update)

                    # Xác nhận cập nhật thành công
                    st.success("Dữ liệu đã được lưu thành công!")  # Thông báo khi lưu thành công
              

                selected_row = grid_response['selected_rows']

                if isinstance(selected_row, pd.DataFrame) :
                    selected_row_df = pd.DataFrame(selected_row)
                    alpha_key = selected_row_df['alpha_key'].iloc[0]
                    author_key = selected_row_df['author_key'].iloc[0]
                if st.button("Xóa"):
                    delete_alpha(conn, alpha_key, author_key)
                    st.experimental_rerun()  
                
            
        elif alpha_action == "Thêm alpha":
            st.sidebar.subheader("Thêm alpha mới")
            new_alpha_key = st.sidebar.text_input("Nhập alpha_key cần thêm")
            authors = fetch_authors(conn)
            selected_author = st.sidebar.selectbox("Chọn tác giả", authors)
            author_key = selected_author if selected_author != "Xem tất cả tác giả" else None
            new_date_start = st.sidebar.date_input("Chọn ngày bắt đầu")
            new_status = st.sidebar.selectbox("Chọn trạng thái", [ "live", "paper", "đã tắt", "wait list", " "])
            new_mark = st.sidebar.selectbox("Chọn đánh dấu", [ "true", "false"])
            new_type = st.sidebar.selectbox("Chọn type", [ "TA", "ML", "Candidate"])
            if st.sidebar.button("Thêm"):
                if new_alpha_key and author_key and new_date_start and new_status and new_mark:
                    if check_alpha_exists(conn, new_alpha_key,author_key):
                        st.error("Alpha key đã tồn tại. Vui lòng nhập alpha_key khác")
                    else:
                        insert_alpha (conn, new_alpha_key, author_key,  new_status, new_mark, new_date_start, new_type)
                        st.success(f"Đã thêm alpha {new_alpha_key}")
                    colnames, pnl_data = fetch_dim_alpha(conn)
                    df = pd.DataFrame(pnl_data, columns=colnames)
                    df = df[~df['alpha_key'].str.startswith('Intraday_')]
                    st.data_editor(df, use_container_width=True)
                
                else:
                    st.error("Vui lòng điền đầy đủ thông tin.")

############################################ QUẢN LÝ DANH MỤC ##################################################

    elif function_choice == 'Quản lý danh mục':
        st.sidebar.header("Quản lý danh mục")
        danhmuc_action = st.sidebar.selectbox("Chọn hành động", [ "Thêm danh mục mới", "Xem và chỉnh sửa", "Xóa danh mục"])
        # Khởi tạo các giá trị mặc định trong session state nếu chưa tồn tại
        if danhmuc_action == "Thêm danh mục mới":
            st.write("Danh sách alpha:")
            colnames, pnl_data = fetch_alpha_fact_pnl(conn)
            df = pd.DataFrame(pnl_data, columns=colnames)
            df = df[~df['alpha_key'].str.startswith('Intraday_')]
            
            # Hiển thị bảng và lấy các hàng đã chọn
            grid_response = hien_thi_bang_du_lieu(df)
            # Lấy các hàng đã chọn
            selected_rows = grid_response.get('selected_rows', [])
           
            if selected_rows is not None: 
                st.write(selected_rows)
                selected_row_df = pd.DataFrame(selected_rows)
                list_name = st.text_input("Nhập tên danh mục:")
                list_author = st.text_input("Nhập tên tác giả:")
                selected_date = st.date_input("Ngày tạo danh mục:", value= datetime.now().date())
                
                # Nút "Lưu"
                if st.button("Lưu", key="save_button"):
                    try:
                        alpha_updates = []
                        for _, row in selected_row_df.iterrows(): 
                            alpha_key = row['alpha_key']
                            author_key = row['author_key']
                            trong_so = row['trong_so']  
                            list_date = datetime.combine(selected_date, time(hour=0, minute=0))
                            add_time = None
                            # Gọi hàm để lưu dữ liệu vào cơ sở dữ liệu
                            insert_new_danhmuc(conn, list_name, list_author, alpha_key, author_key, trong_so, list_date, add_time)
                            update_mark_for_alpha(conn, alpha_key, list_name)
                            alpha_update = {
                                'list_name': list_name,
                                'list_author': list_author,
                                'alpha_key': alpha_key,
                                'author_key': author_key,
                                'trong_so': trong_so,
                                'list_date': list_date,
                                'add_time': None,
                                'del_time': None, 
                                'update_time': None   
                            }
                            alpha_updates.append(alpha_update)
                            
                        update_danhmuc_history1(conn, alpha_updates)
                        st.success("Đã thêm thành công")
                        
                        
                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi: {e}")
            

################################ XEM VÀ CHỈNH SỬA DANH MỤC #####################################################



        elif danhmuc_action == "Xem và chỉnh sửa":
            st.sidebar.header("Lựa chọn chức năng")
            chinhsua_action = st.sidebar.selectbox("Chọn chức năng", ["Xem theo tác giả", "Xem theo danh mục"])
            # Hiển thị các nút hành động
            if chinhsua_action == "Xem theo tác giả":
                authors = fetch_list_author(conn)
                authors.insert(0, "Xem tất cả list_author")
                selected_author = st.sidebar.selectbox("Chọn tác giả", authors)
                list_author = selected_author if selected_author != "Xem tất cả tác giả" else None
                
                df_thong_so_danh_muc = hien_thi_thong_so_danh_muc(conn)
                
                # Giữ lại trạng thái selected_row
                if 'selected_row' not in st.session_state:
                    st.session_state.selected_row = None
                if selected_author == "Xem tất cả list_author":
                    filtered_df = df_thong_so_danh_muc  
                else:
                    filtered_df = df_thong_so_danh_muc[df_thong_so_danh_muc['list_author'] == selected_author]
                grid_response = hien_thi_bang_du_lieu_2(filtered_df)
                # Lấy hàng được chọn
                if grid_response is not None:
                    selected_row = grid_response['selected_rows']
                    if isinstance(selected_row, pd.DataFrame) :
                        selected_row_df = pd.DataFrame(selected_row)
                        st.session_state.selected_row = selected_row_df  # Lưu lại trạng thái

                    # Xử lý khi có hàng được chọn
                    if st.session_state.selected_row is not None:
                        selected_row_df = st.session_state.selected_row

                        if not selected_row_df.empty and 'list_name' in selected_row_df.columns:
                            list_name = selected_row_df['list_name'].iloc[0]
                            list_author = selected_row_df['list_author'].iloc[0]
                            list_date = selected_row_df['list_date'].iloc[0]
                            # Hiển thị dữ liệu PNL cho danh mục đã chọn
                            st.write(f"Dữ liệu PNL cho danh mục: {list_name}")

                            if list_name:
                                list_alphas = fetch_list_alphas(conn, list_name)
                                all_alphas_exist = all(check_alpha_in_fact_positions(conn, alpha['alpha_key']) for alpha in list_alphas)
                                if all_alphas_exist:
                                    colnames, pnl_data = fetch_pnl_list_name_live(conn, list_name)
                                    data_daily = fetch_data(conn, list_name)
                                    plot_multiple_total_gain(data_daily)
                                    # Hiển thị đồ thị
                                    summary_df = tinh_toan_thong_so_live(conn, list_name)
                                    df_results=fetch_and_calculate_weights(conn, list_name)   
                                    plot_multiple_total_gain_history(conn, df_results)
                                    display_df = display_data_vertically(colnames, pnl_data)
                                    # Kết hợp và hiển thị bảng đã chỉnh sửa
                                    if 'alpha_key' in display_df.columns and 'alpha_key' in summary_df.columns:
                                        combined_df = pd.merge(display_df, summary_df, on='alpha_key', how='left')
                                        new_column_order = ['author_key', 'alpha_key','position', 'trong_so', 'profit', 'profit_M', 'profit_3M',
                                                            'margin', 'margin_M', 'margin_3M', 'mdd_score', 'mdd_percent',
                                                            'trad_per_day', 'sharp', 'sharp_M', 'sharp_3M',
                                                            'hit_per_day', 'hitrate', 'hit_M', 'hit_3M', 'return', 'return_M', 'return_3M']
                                        combined_df = combined_df.reindex(columns=new_column_order)
                                        st.write('Danh sách alpha có trong danh mục')
                                        grid_response1 = hien_thi_bang_du_lieu_3(combined_df)
                                        alphas_history = fetch_list_alphas_history(conn, list_name)
                                        data_history = pd.DataFrame(alphas_history)
                                        st.write('Lịch sử chỉnh sửa danh mục')
                                        hien_thi_bang_du_lieu_2(data_history)
                                else:
                                    colnames, pnl_data = fetch_pnl_list_name(conn, list_name)
                                    combined_df = display_data_vertically(colnames, pnl_data)
                                    alphas_history = fetch_list_alphas_history(conn, list_name)
                                    data_history = pd.DataFrame(alphas_history)
                                    data_daily = fetch_data(conn, list_name)
                                    plot_multiple_total_gain(data_daily)
                                    df_results=fetch_and_calculate_weights(conn, list_name)
                                    df_plot = plot_multiple_total_gain_history(conn, df_results)
                                    st.write('Danh sách alpha có trong danh mục')
                                    grid_response1 = hien_thi_bang_du_lieu_3(combined_df)
                                    st.write('Lịch sử chỉnh sửa danh mục')
                                    hien_thi_bang_du_lieu_2(data_history)
                                   

                                # Nếu nhấn "Cập nhật"
                                # Nếu nhấn "Thêm mới"
                                if 'Update' not in st.session_state:
                                    st.session_state.Update = False
                                if 'Save' not in st.session_state:  # Thêm biến Save để điều khiển hiển thị nút Lưu
                                    st.session_state.Save = False

                                # Khi nhấn nút "Cập nhật", chuyển trạng thái Update thành True và hiển thị trường chọn ngày
                                if st.button("Cập nhật"):
                                    st.session_state.Update = True
                                    st.session_state.Save = False  
                                # Nếu Update là True, hiển thị bảng và cho phép chọn hàng
                                if st.session_state.Update:
                                    edited_df = pd.DataFrame(grid_response1['data'])
                                    merged_df = pd.merge(combined_df, edited_df, how='outer', indicator=True)
                                    changed_rows = merged_df[merged_df['_merge'] == 'right_only']
                                    data_to_update = []
                                    history_updates = []

                                    # Hiển thị trường chọn ngày
                                    selected_date = st.date_input("Chọn ngày cập nhật", value=datetime.now().date())
                                    update_time = datetime.combine(selected_date, time(hour=0, minute=0))

                                    # Hiển thị nút Lưu
                                    if st.button("Lưu"):  # Khi nhấn Lưu, bắt đầu quá trình lưu
                                        st.session_state.Save = True

                                    # Nếu đã nhấn nút Lưu, xử lý cập nhật dữ liệu
                                    if st.session_state.Save:
                                        for index, row in changed_rows.iterrows():
                                            existing_history = get_alpha_history(conn, list_name, row['alpha_key'], row['author_key'])
                                            updated_row = (
                                                row['trong_so'], 
                                                row['alpha_key'],  
                                                row['author_key'], 
                                                list_name         
                                            )
                                            data_to_update.append(updated_row)
                                            if not existing_history.empty:
                                                existing_history_sorted = existing_history.sort_values(by='add_time', ascending=False)
                                                add_time = existing_history_sorted['add_time'].iloc[0]
                                                history_update = {
                                                    'list_name': list_name,
                                                    'list_author': list_author,
                                                    'alpha_key': row['alpha_key'],
                                                    'author_key': row['author_key'],
                                                    'trong_so': row['trong_so'],            
                                                    'list_date': list_date,
                                                    'add_time': add_time,
                                                    'del_time': None,
                                                    'update_time': update_time
                                                }
                                                history_updates.append(history_update)
                                                                                
                                        
                                        if data_to_update:
                                            update_trong_so_bulk1(conn, data_to_update)
                                        if history_updates:
                                            update_danhmuc_history1(conn, history_updates)
                                        st.session_state.Update = False
                                        st.session_state.Save = False
                                        st.session_state.reload = True
                                        st.session_state.selected_row = selected_row_df
                                        st.experimental_rerun()
                                    
                                    
                                #### NÚT THÊM MỚI
                                
                                # Nếu nhấn "Thêm mới"
                                if 'show_add_alpha' not in st.session_state:
                                    st.session_state.show_add_alpha = False

                                # Khi nhấn nút "Thêm alpha", chuyển trạng thái show_add_alpha thành True
                                if st.button("Thêm alpha"):
                                    st.session_state.show_add_alpha = True

                                # Nếu show_add_alpha là True, hiển thị bảng và cho phép chọn hàng
                                if st.session_state.show_add_alpha:
                                    colnames, pnl_data = fetch_alpha_fact_pnl(conn)
                                    df = pd.DataFrame(pnl_data, columns=colnames)
                                    df = df[~df['alpha_key'].str.startswith('Intraday_')]
                                    # Hiển thị bảng dữ liệu với key cố định để tái tạo bảng
                                    grid_response_insert = hien_thi_bang_du_lieu(df)
                                    selected_rows_insert =  grid_response_insert.get('selected_rows', [])
                                    if selected_rows_insert is not None and len(selected_rows_insert) > 0:  # Kiểm tra xem có hàng nào được chọn hay không
                                        selected_row_them_df = pd.DataFrame(selected_rows_insert)
                                        if not selected_row_them_df.empty:
                                            selected_date = st.date_input("Chọn ngày thêm alpha: ", value= datetime.now().date())
                                            # Nút "Lưu"
                                            if st.button("Lưu", key="save_button"):
                                                try:
                                                    for _, row in selected_row_them_df.iterrows():  # Lặp qua từng dòng được chọn
                                                        alpha_key = row['alpha_key']
                                                        author_key = row['author_key']
                                                        trong_so = row['trong_so']  # Lấy giá trị trọng số
                                                        add_time = datetime.combine(selected_date, time(hour=0, minute=0))
                                                        # Gọi hàm để lưu dữ liệu vào cơ sở dữ liệu
                                                        insert_new_danhmuc(conn, list_name, list_author,  alpha_key, author_key, trong_so, list_date, add_time)
                                                        update_mark_for_alpha(conn, alpha_key, list_name)
                                                        # Cập nhật bảng danh mục lịch sử
                                                        alpha_updates = [{
                                                            'list_name': list_name,
                                                            'list_author': list_author,
                                                            'alpha_key': alpha_key,
                                                            'author_key': author_key,
                                                            'trong_so': trong_so,
                                                            'list_date': list_date,
                                                            'add_time': add_time,
                                                            'del_time': None,
                                                            'update_date': None
                                                        }]
                                                        update_danhmuc_history1(conn, alpha_updates)
                                                        
                                                    st.session_state.show_add_alpha = False
                                                    st.session_state.reload = True
                                                    st.session_state.selected_row = selected_row_df
                                                    st.experimental_rerun()
                                                    
                                                except Exception as e:
                                                    st.error(f"Đã xảy ra lỗi: {e}")
                                

                                ####### XÓA 1 ALPHA TRONG DANH MỤC #######

                                selected_rows1 = grid_response1['selected_rows']
                                if isinstance(selected_rows1, pd.DataFrame) and len(selected_rows1) > 0:
                                    selected_row_df1 =  pd.DataFrame(selected_rows1)
                                    if not selected_row_df1.empty:
                                        selected_date = st.date_input("Chọn ngày xóa alpha: ", value= datetime.now().date())
                                        # Nút "Xóa"
                                        if st.button("Xóa"):
                                            alpha_key = selected_row_df1['alpha_key'].iloc[0]
                                            author_key = selected_row_df1['author_key'].iloc[0]
                                            trong_so = selected_row_df1['trong_so'].iloc[0] 
                                            del_time = datetime.combine(selected_date, time(hour=0, minute=0))

                                            existing_history = get_alpha_history(conn, list_name, alpha_key, author_key)
                                            if not existing_history.empty:
                                                existing_history_sorted = existing_history.sort_values(by='add_time', ascending=False)
                                                # Lấy giá trị add_time và ngày cập nhật gần nhất
                                                add_time = existing_history_sorted['add_time'].iloc[0]  # Bản ghi thêm vào đầu tiên
                                                last_update_date = existing_history_sorted['update_time'].dropna().iloc[-1] if not existing_history_sorted['update_time'].dropna().empty else None

                                                alpha_updates = [{
                                                                'list_name': list_name,
                                                                'list_author': list_author,
                                                                'alpha_key': alpha_key,
                                                                'author_key': author_key,
                                                                'trong_so': trong_so,
                                                                'list_date': list_date,
                                                                'add_time': add_time,
                                                                'del_time': del_time,
                                                                'update_time': last_update_date
                                                            }]
                                                update_danhmuc_history1(conn, alpha_updates)
                                                delete_danhmuc_alpha(conn, list_name, alpha_key, author_key)
                                                st.session_state.reload = True
                                                st.session_state.selected_row  = selected_row_df
                                                st.experimental_rerun()
                        if 'reload' in st.session_state and st.session_state.reload:
                            st.session_state.reload = False
                            st.experimental_rerun()
            else:
                list_01 = fetch_list_name(conn)
                list_01.insert(0, "Xem tất cả danh mục")
                selected_list = st.sidebar.selectbox("Chọn danh mục", list_01)
                list_name = selected_list if selected_list != "Xem tất cả danh mục" else None
                df_thong_so_danh_muc = hien_thi_thong_so_danh_muc(conn)
                
                # Giữ lại trạng thái selected_row
                if 'selected_row' not in st.session_state:
                    st.session_state.selected_row = None
                if selected_list == "Xem tất cả danh mục":
                    filtered_df = df_thong_so_danh_muc  
                else:
                    filtered_df = df_thong_so_danh_muc[df_thong_so_danh_muc['list_name'] == selected_list]
                grid_response = hien_thi_bang_du_lieu_2(filtered_df)
                # Lấy hàng được chọn
                selected_row = grid_response['selected_rows']
                if isinstance(selected_row, pd.DataFrame) and len(selected_row) > 0:
                    selected_row_df = pd.DataFrame(selected_row)
                    st.session_state.selected_row = selected_row_df  # Lưu lại trạng thái

                # Xử lý khi có hàng được chọn
                if st.session_state.selected_row is not None:
                    selected_row_df = st.session_state.selected_row
                    if not selected_row_df.empty and 'list_name' in selected_row_df.columns:
                        list_name = selected_row_df['list_name'].iloc[0]
                        list_author = selected_row_df['list_author'].iloc[0]
                        # Hiển thị dữ liệu PNL cho danh mục đã chọn
                        st.write(f"Dữ liệu PNL cho danh mục: {list_name}")

                        if list_name:
                            list_alphas = fetch_list_alphas(conn, list_name)
                            all_alphas_exist = all(check_alpha_in_fact_positions(conn, alpha['alpha_key']) for alpha in list_alphas)
                            if all_alphas_exist:
                                colnames, pnl_data = fetch_pnl_list_name_live(conn, list_name)
                                display_df = display_data_vertically(colnames, pnl_data)

                                # Hiển thị đồ thị
                                summary_df = tinh_toan_thong_so_live(conn, list_name)
                                data_daily = fetch_data(conn, list_name)
                                plot_multiple_total_gain(data_daily)
                                df_results=fetch_and_calculate_weights(conn, list_name)
                                plot_multiple_total_gain_history(conn, df_results)

                                # Kết hợp và hiển thị bảng đã chỉnh sửa
                                if 'alpha_key' in display_df.columns and 'alpha_key' in summary_df.columns:
                                    combined_df = pd.merge(display_df, summary_df, on='alpha_key', how='left')
                                    new_column_order = ['author_key', 'alpha_key','position', 'trong_so', 'profit', 'profit_M', 'profit_3M',
                                                        'margin', 'margin_M', 'margin_3M', 'mdd_score', 'mdd_percent',
                                                        'trad_per_day', 'sharp', 'sharp_M', 'sharp_3M',
                                                        'hit_per_day', 'hitrate', 'hit_M', 'hit_3M', 'return', 'return_M', 'return_3M']
                                    combined_df = combined_df.reindex(columns=new_column_order)
                                    grid_response1 = hien_thi_bang_du_lieu_3(combined_df)
                                    alphas_history = fetch_list_alphas_history(conn, list_name)
                                    data_history = pd.DataFrame(alphas_history)
                                    st.write('Lịch sử chỉnh sửa danh mục')
                                    hien_thi_bang_du_lieu_2(data_history)
                                
                            else:
                                colnames, pnl_data = fetch_pnl_list_name(conn, list_name)
                                combined_df = display_data_vertically(colnames, pnl_data)
                                alphas_history = fetch_list_alphas_history(conn, list_name)
                                data_history = pd.DataFrame(alphas_history)
                                data_daily = fetch_data(conn, list_name)
                                plot_multiple_total_gain(conn,  data_daily)
                                df_results=fetch_and_calculate_weights(conn, list_name)
                                plot_multiple_total_gain_history(conn, df_results)
                                st.write('Danh sách alpha có trong danh mục')
                                grid_response1 = hien_thi_bang_du_lieu_3(combined_df)
                                st.write('Lịch sử chỉnh sửa danh mục')
                                hien_thi_bang_du_lieu_2(data_history)
                            # Nếu nhấn "Cập nhật"
                            if 'Update' not in st.session_state:
                                st.session_state.Update = False
                            if 'Save' not in st.session_state:  # Thêm biến Save để điều khiển hiển thị nút Lưu
                                st.session_state.Save = False

                            # Khi nhấn nút "Cập nhật", chuyển trạng thái Update thành True và hiển thị trường chọn ngày
                            if st.button("Cập nhật"):
                                st.session_state.Update = True
                                st.session_state.Save = False  
                            # Nếu Update là True, hiển thị bảng và cho phép chọn hàng
                            if st.session_state.Update:
                                edited_df = pd.DataFrame(grid_response1['data'])
                                merged_df = pd.merge(combined_df, edited_df, how='outer', indicator=True)
                                changed_rows = merged_df[merged_df['_merge'] == 'right_only']
                                data_to_update = []
                                history_updates = []

                                # Hiển thị trường chọn ngày
                                selected_date = st.date_input("Chọn ngày cập nhật", value=datetime.now().date())
                                update_time = datetime.combine(selected_date, time(hour=0, minute=0))

                                # Hiển thị nút Lưu
                                if st.button("Lưu"):  # Khi nhấn Lưu, bắt đầu quá trình lưu
                                    st.session_state.Save = True

                                # Nếu đã nhấn nút Lưu, xử lý cập nhật dữ liệu
                                if st.session_state.Save:
                                    for index, row in changed_rows.iterrows():
                                        existing_history = get_alpha_history(conn, list_name, row['alpha_key'], row['author_key'])
                                        updated_row = (
                                            row['trong_so'], 
                                            row['alpha_key'],  
                                            row['author_key'], 
                                            list_name         
                                        )
                                        data_to_update.append(updated_row)
                                        if not existing_history.empty:
                                            existing_history_sorted = existing_history.sort_values(by='add_time', ascending=False)
                                            add_time = existing_history_sorted['add_time'].iloc[0]
                                            history_update = {
                                                'list_name': list_name,
                                                'list_author': list_author,
                                                'alpha_key': row['alpha_key'],
                                                'author_key': row['author_key'],
                                                'trong_so': row['trong_so'],            
                                                'list_date': list_date,
                                                'add_time': add_time,
                                                'del_time': None,
                                                'update_time': update_time
                                            }
                                            history_updates.append(history_update)
                                                                            
                                    
                                    if data_to_update:
                                        update_trong_so_bulk1(conn, data_to_update)
                                    if history_updates:
                                        update_danhmuc_history1(conn, history_updates)
                                    st.session_state.Update = False
                                    st.session_state.Save = False
                                    st.session_state.reload = True
                                    st.session_state.selected_row = selected_row_df
                                    st.experimental_rerun()
                            # Xóa 1 alpha trong danh mục
                            selected_rows1 = grid_response1['selected_rows']
                            if isinstance(selected_rows1, pd.DataFrame) and len(selected_rows1) > 0:
                                selected_row_df1 =  pd.DataFrame(selected_rows1)
                                if not selected_row_df1.empty:
                                    selected_date = st.date_input("Chọn ngày xóa alpha: ", value=pd.Timestamp.now().date())
                                    # Nút "Xóa"
                                    if st.button("Xóa"):
                                            alpha_key = selected_row_df1['alpha_key'].iloc[0]
                                            author_key = selected_row_df1['author_key'].iloc[0]
                                            trong_so = selected_row_df1['trong_so'].iloc[0] 
                                            del_time = datetime.combine(selected_date, time(hour=0, minute=0))

                                            existing_history = get_alpha_history(conn, list_name, alpha_key, author_key)
                                            if not existing_history.empty:
                                                existing_history_sorted = existing_history.sort_values(by='add_time', ascending=False)
                                                # Lấy giá trị add_time và ngày cập nhật gần nhất
                                                add_time = existing_history_sorted['add_time'].iloc[0]  # Bản ghi thêm vào đầu tiên
                                                last_update_date = existing_history_sorted['update_time'].dropna().iloc[-1] if not existing_history_sorted['update_time'].dropna().empty else None

                                                alpha_updates = [{
                                                                'list_name': list_name,
                                                                'list_author': list_author,
                                                                'alpha_key': alpha_key,
                                                                'author_key': author_key,
                                                                'trong_so': trong_so,
                                                                'list_date': list_date,
                                                                'add_time': add_time,
                                                                'del_time': del_time,
                                                                'update_time': last_update_date
                                                            }]
                                                update_danhmuc_history1(conn, alpha_updates)
                                                delete_danhmuc_alpha(conn, list_name, alpha_key, author_key)
                                                st.session_state.reload = True
                                                st.session_state.selected_row  = selected_row_df
                                                st.experimental_rerun()
                                # Nếu nhấn "Thêm mới"
                                if 'show_add_alpha' not in st.session_state:
                                    st.session_state.show_add_alpha = False

                                # Khi nhấn nút "Thêm alpha", chuyển trạng thái show_add_alpha thành True
                                if st.button("Thêm alpha"):
                                    st.session_state.show_add_alpha = True

                                # Nếu show_add_alpha là True, hiển thị bảng và cho phép chọn hàng
                                if st.session_state.show_add_alpha:
                                        colnames, pnl_data = fetch_alpha_fact_pnl(conn)
                                        df = pd.DataFrame(pnl_data, columns=colnames)
                                        df = df[~df['alpha_key'].str.startswith('Intraday_')]
                                        # Hiển thị bảng dữ liệu với key cố định để tái tạo bảng
                                        grid_response_insert = hien_thi_bang_du_lieu(df)
                                        selected_rows_insert =  grid_response_insert.get('selected_rows', [])
                                        if selected_rows_insert is not None and len(selected_rows_insert) > 0:  # Kiểm tra xem có hàng nào được chọn hay không
                                            selected_row_them_df = pd.DataFrame(selected_rows_insert)
                                            if not selected_row_them_df.empty:
                                                selected_date = st.date_input("Chọn ngày thêm alpha: ", value= datetime.now().date())
                                                # Nút "Lưu"
                                                if st.button("Lưu", key="save_button"):
                                                    try:
                                                        for _, row in selected_row_them_df.iterrows():  # Lặp qua từng dòng được chọn
                                                            alpha_key = row['alpha_key']
                                                            author_key = row['author_key']
                                                            trong_so = row['trong_so']  # Lấy giá trị trọng số
                                                            add_time = datetime.combine(selected_date, time(hour=0, minute=0))
                                                            # Gọi hàm để lưu dữ liệu vào cơ sở dữ liệu
                                                            insert_new_danhmuc(conn, list_name, list_author,  alpha_key, author_key, trong_so, list_date, add_time)
                                                            update_mark_for_alpha(conn, alpha_key, list_name)
                                                            # Cập nhật bảng danh mục lịch sử
                                                            alpha_updates = [{
                                                                'list_name': list_name,
                                                                'list_author': list_author,
                                                                'alpha_key': alpha_key,
                                                                'author_key': author_key,
                                                                'trong_so': trong_so,
                                                                'list_date': list_date,
                                                                'add_time': add_time,
                                                                'del_time': None,
                                                                'update_date': None
                                                            }]
                                                            update_danhmuc_history1(conn, alpha_updates)
                                                            
                                                        st.session_state.show_add_alpha = False
                                                        st.session_state.reload = True
                                                        st.session_state.selected_row = selected_row_df
                                                        st.experimental_rerun()
                                                        
                                                    except Exception as e:
                                                        st.error(f"Đã xảy ra lỗi: {e}")

                                
                                if 'reload' in st.session_state and st.session_state.reload:
                                    st.session_state.reload = False
                                    st.experimental_rerun()
        else:
            df_thong_so_danh_muc = hien_thi_thong_so_danh_muc(conn)     
            grid_response = hien_thi_bang_du_lieu_2(df_thong_so_danh_muc)
            # Lấy hàng được chọn
            if grid_response is not None:
                if 'selected_row' not in st.session_state:
                    st.session_state.selected_row = None
                selected_row = grid_response['selected_rows']
                if isinstance(selected_row, pd.DataFrame) and len(selected_row) > 0:
                    selected_row_df = pd.DataFrame(selected_row)
                    st.session_state.selected_row = selected_row_df  # Lưu lại trạng thái

                # Xử lý khi có hàng được chọn
                if st.session_state.selected_row is not None:
                    selected_row_df = st.session_state.selected_row
                    if st.button('Xóa'):
                        if not selected_row_df.empty and 'list_name' in selected_row_df.columns:
                            list_name = selected_row_df['list_name'].iloc[0]
                            list_author = selected_row_df['list_author'].iloc[0]
                            # Lấy danh sách alpha_key từ bảng danhmuc_alpha
                            alpha_keys = get_alpha_keys_for_list(conn, list_name, list_author)
                            # Cập nhật mark trong bảng dim_alpha
                            update_mark_after_delete(conn, alpha_keys, list_name)

                            # Xóa danh mục trong bảng danhmuc_alpha và cập nhật lịch sử
                            del_danhmuc(conn, list_name, list_author)
                            del_danhmuc_history(conn, list_name, list_author)
                            st.experimental_rerun()
                            
                        
                else:
                    st.warning('Vui lòng chọn danh mục cần xóa')
                    
                
        conn.close()

if __name__ == "__main__":
    main()