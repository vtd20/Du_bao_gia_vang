# danh_gia_mo_hinh
from pyspark.sql import SparkSession
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import pandas as pd

# Tạo đối tượng Spark
spark = SparkSession.builder.appName("GoldPricePrediction").getOrCreate()

def evaluate_model(model, X_test, y_test, scaler_model, spark):
    # Tải mô hình tốt nhất
    best_model = load_model('best_model.keras')
    
    # Dự đoán bằng mô hình đã tải
    y_pred = best_model.predict(X_test)
    
    # Đánh giá mô hình
    result = model.evaluate(X_test, y_test)  # Giả định bạn có một mô hình Keras cho việc này
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    Accuracy = 1 - MAPE
    
    print("Mất mát trên tập kiểm tra:", result)
    print("MAPE trên tập kiểm tra:", MAPE)
    print("Độ chính xác trên tập kiểm tra:", Accuracy)
    print("R-squared:", r2)
    
    # Nghịch đảo chuẩn hóa dự đoán để vẽ biểu đồ
    to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    
    # Chuyển đổi y_test và y_pred thành Spark DataFrames
    y_test_df = spark.createDataFrame(pd.DataFrame(y_test, columns=['features']))
    y_pred_df = spark.createDataFrame(pd.DataFrame(y_pred, columns=['features']))
    
    # Áp dụng nghịch đảo chuẩn hóa bằng scaler_model
    y_test_scaled = scaler_model.transform(y_test_df.withColumn("features", to_vector_udf("features"))).select("features")
    y_pred_scaled = scaler_model.transform(y_pred_df.withColumn("features", to_vector_udf("features"))).select("features")
    
    # Thu thập kết quả
    y_test_true = y_test_scaled.collect()
    y_test_pred = y_pred_scaled.collect()
    
    # Trích xuất giá trị từ các cột features
    y_test_true_values = [row['features'][0] for row in y_test_true]  # Lấy giá trị thực tế
    y_test_pred_values = [row['features'][0] for row in y_test_pred]  # Lấy giá trị dự đoán
    
    return y_test_true_values, y_test_pred_values

def main():
    # Đảm bảo spark đã được khởi tạo trước khi gọi evaluate_model
    # Các bước khác của chương trình...

    # Gọi hàm evaluate_model và truyền spark vào
    y_test_true_values, y_test_pred_values = evaluate_model(model, X_test, y_test, scaler_model, spark)

if __name__ == "__main__":
    main()
