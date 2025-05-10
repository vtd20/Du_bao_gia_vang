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
    # Load best model
    best_model = load_model('best_model.keras')
    
    # Predict using the loaded model
    y_pred = best_model.predict(X_test)
    
    # Evaluate the model
    result = model.evaluate(X_test, y_test)  # Assuming you have a Keras model for this
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    Accuracy = 1 - MAPE
    
    print("Test Loss:", result)
    print("Test MAPE:", MAPE)
    print("Test Accuracy:", Accuracy)
    print("R-squared:", r2)
    
    # Inverse scale predictions for plotting
    to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    
    # Convert y_test and y_pred to Spark DataFrames
    y_test_df = spark.createDataFrame(pd.DataFrame(y_test, columns=['features']))
    y_pred_df = spark.createDataFrame(pd.DataFrame(y_pred, columns=['features']))
    
    # Apply inverse scaling using scaler_model
    y_test_scaled = scaler_model.transform(y_test_df.withColumn("features", to_vector_udf("features"))).select("features")
    y_pred_scaled = scaler_model.transform(y_pred_df.withColumn("features", to_vector_udf("features"))).select("features")
    
    # Collect results
    y_test_true = y_test_scaled.collect()
    y_test_pred = y_pred_scaled.collect()
    
    # Extract the values from the features columns
    y_test_true_values = [row['features'][0] for row in y_test_true]  # Get actual true values
    y_test_pred_values = [row['features'][0] for row in y_test_pred]  # Get predicted values
    
    return y_test_true_values, y_test_pred_values

def main():
    # Đảm bảo spark đã được khởi tạo trước khi gọi evaluate_model
    # Các bước khác của chương trình...

    # Gọi hàm evaluate_model và truyền spark vào
    y_test_true_values, y_test_pred_values = evaluate_model(model, X_test, y_test, scaler_model, spark)

if __name__ == "__main__":
    main()


