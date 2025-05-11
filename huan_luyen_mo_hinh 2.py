# huan_luyen_mo_hinh
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam  # Hoặc Nadam, nếu bạn muốn kiểm soát rõ ràng

def define_model(window_size):
    # Định nghĩa đầu vào với kích thước cửa sổ
    input1 = Input(shape=(window_size, 1))

    # Lớp LSTM đầu tiên với 128 đơn vị, trả về chuỗi
    x = LSTM(units=128, return_sequences=True)(input1)

    # Áp dụng Dropout để tránh overfitting
    x = Dropout(0.3)(x)

    # Lớp LSTM thứ hai với 64 đơn vị, trả về chuỗi
    x = LSTM(units=64, return_sequences=True)(x)

    # Áp dụng Dropout
    x = Dropout(0.3)(x)

    # Lớp LSTM thứ ba với 64 đơn vị
    x = LSTM(units=64)(x)

    # Áp dụng Dropout
    x = Dropout(0.3)(x)

    # Lớp Dense với 32 đơn vị và hàm kích hoạt ReLU
    x = Dense(32, activation='relu')(x)  # Thay đổi từ 'softmax' sang 'relu'

    # Lớp đầu ra với 1 đơn vị
    dnn_output = Dense(1)(x)

    # Tạo mô hình
    model = Model(inputs=input1, outputs=[dnn_output])

    # Có thể đặt rõ ràng tỷ lệ học tại đây nếu sử dụng Adam
    optimizer = Adam(learning_rate=0.0001)  # Tỷ lệ học ví dụ

    # Biên dịch mô hình với hàm mất mát là mean squared error
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Hiển thị tóm tắt mô hình
    model.summary()

    return model

def train_model(model, X_train, y_train):
    # Lưu mô hình tốt nhất dựa trên giá trị mất mát trên tập kiểm tra
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    # Dừng sớm nếu giá trị mất mát trên tập kiểm tra không cải thiện
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Thêm callback ReduceLROnPlateau để giảm tỷ lệ học khi mất mát không cải thiện
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Huấn luyện mô hình
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[checkpoint, early_stopping, reduce_lr])

    # Vẽ biểu đồ mất mát huấn luyện và kiểm tra
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Mất mát huấn luyện')
    plt.plot(history.history['val_loss'], label='Mất mát kiểm tra')
    plt.title('Mất mát huấn luyện và kiểm tra')
    plt.xlabel('Vòng lặp (Epoch)')
    plt.ylabel('Mất mát')
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")
    plt.close()

    return model, history
