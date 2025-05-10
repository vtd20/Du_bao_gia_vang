# xu_ly_va_tai_du_lieu
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, asc, monotonically_increasing_id, row_number, regexp_replace, col, count, countDistinct, isnan, when
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType

def load_and_preprocess_data(file_path, hdfs_output_path):
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("GoldPricePrediction") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()
    
    # Đọc dữ liệu từ file CSV
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df.show(10)
    
    # Kiểm tra số lượng dòng và cột
    num_rows = df.count()
    num_cols = len(df.columns)
    print("# Số lượng hàng và cột:\n")
    print(f"Shape: ({num_rows}, {num_cols})")
    
    # Xóa các cột không cần thiết
    df = df.drop('Vol.', 'Change %')
    
    # In cấu trúc schema của DataFrame
    print("# Cấu trúc Schema của DataFrame:\n")
    df.printSchema()
    
    # Chuyển đổi cột 'Date' sang kiểu ngày tháng và sắp xếp theo ngày
    df = df.withColumn('Date', to_date(df['Date'], 'MM/dd/yyyy')).orderBy(asc('Date'))
    
    # Đặt lại chỉ số
    df = df.withColumn("index", monotonically_increasing_id()) \
           .withColumn("row_num", row_number().over(Window.orderBy("index"))) \
           .drop("index") \
           .withColumnRenamed("row_num", "index") \
           .orderBy(asc("index"))
    
    # Chuyển các cột số thành DoubleType và loại bỏ dấu phẩy
    NumCols = [c for c in df.columns if c != 'Date']
    for column in NumCols:
        df = df.withColumn(column, regexp_replace(col(column), ',', '').cast(DoubleType()))
    
    # Kiểm tra dữ liệu trùng lặp
    total_count = df.count()
    distinct_count = df.distinct().count()
    duplicate_count = total_count - distinct_count
    print("# Số lượng hàng trùng lặp:\n")
    print("Số lượng hàng trùng lặp: ", duplicate_count)
    
    # Kiểm tra giá trị NaN hoặc null trong các cột
    print("# Kiểm tra giá trị NaN hoặc null trong các cột: \n")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns if c != 'Date']).show()
    df.show()
    
    # Lưu dữ liệu đã xử lý vào HDFS dưới dạng Parquet
    df.write.mode("overwrite").parquet(hdfs_output_path)
    print(f"Data saved to HDFS at: {hdfs_output_path}")
    
    return spark, df

