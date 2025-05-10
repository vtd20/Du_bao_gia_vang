# huan_luyen_mo_hinh
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam  # Or Nadam, if you prefer explicit control
 
def define_model(window_size):

    input1 = Input(shape=(window_size, 1))

    x = LSTM(units=128, return_sequences=True)(input1)

    x = Dropout(0.3)(x)

    x = LSTM(units=64, return_sequences=True)(x)

    x = Dropout(0.3)(x)

    x = LSTM(units=64)(x)

    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)  # Changed from 'softmax' to 'relu'

    dnn_output = Dense(1)(x)
 
    model = Model(inputs=input1, outputs=[dnn_output])

    # You can explicitly set the learning rate here if using Adam

    optimizer = Adam(learning_rate=0.0001) # Example learning rate

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
 
    return model
 
def train_model(model, X_train, y_train):

    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Add ReduceLROnPlateau callback

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[checkpoint, early_stopping, reduce_lr])
 
    # Plot training and validation loss

    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='Training Loss')

    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.grid()

    plt.savefig("loss.png")
    plt.close()
 
    return model, history
 
