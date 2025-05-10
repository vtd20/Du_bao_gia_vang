# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import year
from xu_ly_va_tai_du_lieu import load_and_preprocess_data
from truc_quan_hoa import plot_gold_price_history, plot_train_test_split, plot_model_performance
from chuan_hoa_du_lieu import scale_and_prepare_data
from huan_luyen_mo_hinh import define_model, train_model
from danh_gia_mo_hinh import evaluate_model

def main():
    # Khởi tạo SparkSession (chỉ cần một lần duy nhất)
    spark = SparkSession.builder \
        .appName("GoldPricePrediction") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()

    # Đường dẫn file
    csv_file_path = 'hdfs://localhost:9000/gold_price_data/gold_price_2013_2023.csv'
    hdfs_input_path = 'hdfs://localhost:9000/gold_price_data/processed_data'
    
    # Tải và xử lý dữ liệu, lưu vào HDFS
    spark, df = load_and_preprocess_data(csv_file_path, hdfs_input_path)
    
    # Đọc lại dữ liệu đã xử lý từ HDFS
    df = spark.read.parquet(hdfs_input_path)
    
    # Hiển thị lịch sử giá vàng
    plot_gold_price_history(df)
    
    # Tính toán kích thước bộ kiểm tra
    test_size = df.filter(year(df.Date) == 2021).count()
    
    # Hiển thị biểu đồ phân chia bộ huấn luyện và kiểm tra
    plot_train_test_split(df, test_size)
    
    # Chuẩn hóa và chuẩn bị dữ liệu
    window_size = 30
    scaler_model, X_train, y_train, X_test, y_test, train_data = scale_and_prepare_data(df, test_size, window_size)
    
    # Định nghĩa và huấn luyện mô hình
    model = define_model(window_size)
    model, history = train_model(model, X_train, y_train)
    
    # Đánh giá mô hình
    y_test_true_values, y_test_pred_values = evaluate_model(model, X_test, y_test, scaler_model, spark)
    
    # Hiển thị hiệu suất mô hình
    plot_model_performance(df, train_data, y_test_true_values, y_test_pred_values, test_size)
    
    # Dừng SparkSession sau khi hoàn thành
    spark.stop()

if __name__ == "__main__":
    main()

