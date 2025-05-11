# truc_quan_hoa
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.functions import year

def plot_gold_price_history(df):
    # Chuyển đổi thành Pandas DataFrame
    pdf = df.toPandas()
    
    # Vẽ biểu đồ đường bằng Plotly
    fig = px.line(pdf, x='Date', y='Price')
    fig.update_traces(line_color='blue')
    fig.update_layout(
        xaxis_title="Ngày",
        yaxis_title="Giá đã chuẩn hóa",
        title={'text': "Lịch sử giá vàng", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
        plot_bgcolor='rgba(255,223,0,0.8)'
    )
    
    # Nếu đang chạy trong môi trường tương tác, hiển thị biểu đồ; nếu không, lưu dưới dạng HTML
    try:
        fig.show()  # Nếu bạn đang ở trong Jupyter Notebook hoặc môi trường tương tác khác
    except:
        fig.write_html("gold_price_history.html")  # Lưu dưới dạng HTML cho môi trường không tương tác

def plot_train_test_split(df, test_size):
    # Chuyển đổi thành Pandas DataFrame
    pdf = df.toPandas()
    
    # Vẽ biểu đồ phân chia tập huấn luyện/kiểm tra bằng Matplotlib
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    
    plt.plot(pdf['Date'][:-test_size], pdf['Price'][:-test_size], color='blue', lw=2)
    plt.plot(pdf['Date'][-test_size:], pdf['Price'][-test_size:], color='red', lw=2)
    
    plt.title('Tập huấn luyện và kiểm tra giá vàng', fontsize=15)
    plt.xlabel('Ngày', fontsize=12)
    plt.ylabel('Giá', fontsize=12)
    plt.legend(['Tập huấn luyện', 'Tập kiểm tra'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    
    # Lưu biểu đồ dưới dạng PNG nếu không ở trong môi trường tương tác
    plt.savefig("train_test_split.png")
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ

def plot_model_performance(df, train_data, y_test_true, y_test_pred, test_size):
    # Chuyển đổi thành Pandas DataFrame
    pdf = df.toPandas()
    
    plt.figure(figsize=(15, 6), dpi=150)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('axes', edgecolor='black')
    
    plt.plot(pdf['Date'].iloc[:-test_size], train_data, color='green', lw=2)
    plt.plot(pdf['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2)
    plt.plot(pdf['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2)
    
    plt.title('Hiệu suất mô hình trong dự đoán giá vàng', fontsize=15)
    plt.xlabel('Ngày', fontsize=12)
    plt.ylabel('Giá', fontsize=12)
    plt.legend(['Dữ liệu huấn luyện', 'Dữ liệu kiểm tra thực tế', 'Dữ liệu kiểm tra dự đoán'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    
    # Lưu biểu đồ dưới dạng PNG nếu không ở trong môi trường tương tác
    plt.savefig("model_performance.png")
    plt.close()  # Đóng biểu đồ để giải phóng bộ nhớ
