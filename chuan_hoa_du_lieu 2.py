# chuan_hoa_du_lieu
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col
import numpy as np

def scale_and_prepare_data(df, test_size, window_size=30):
    # Define UDF to convert double to vector
    to_vector_udf = udf(lambda x: Vectors.dense([x]), VectorUDT())
    
    # Prepare data for scaling
    price_df = df.withColumn("features", to_vector_udf("Price"))
    
    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(price_df)
    scaled_df = scaler_model.transform(price_df)
    
    # Training set
    train_data_df = df.filter(col("index") <= df.count() - test_size).select("Price", "index", "Date")
    train_data_df = train_data_df.withColumn("features", to_vector_udf("Price"))
    train_data_scaled_df = scaler_model.transform(train_data_df)
    train_data = np.array(train_data_scaled_df.select("scaled_features").collect()).reshape(-1, 1)
    
    X_train = [train_data[i-window_size:i, 0] for i in range(window_size, len(train_data))]
    y_train = [train_data[i, 0] for i in range(window_size, len(train_data))]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Test set
    test_data_df = df.filter(col("index") > df.count() - test_size - window_size).select("Price", "index", "Date")
    test_data_df = test_data_df.withColumn("features", to_vector_udf("Price"))
    test_data_scaled_df = scaler_model.transform(test_data_df)
    test_data = np.array(test_data_scaled_df.select("scaled_features").collect()).reshape(-1, 1)
    
    X_test = [test_data[i-window_size:i, 0] for i in range(window_size, len(test_data))]
    y_test = [test_data[i, 0] for i in range(window_size, len(test_data))]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    
    print('X_train Shape: ', X_train.shape)
    print('y_train Shape: ', y_train.shape)
    print('X_test Shape:  ', X_test.shape)
    print('y_test Shape:  ', y_test.shape)
    
    return scaler_model, X_train, y_train, X_test, y_test, train_data
