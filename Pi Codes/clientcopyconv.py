import flwr as fl
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError 
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError


# Load model and data (MobileNetV2, CIFAR-10)
# model = tf.keras.applications.MobileNetV3Large((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
df = pd.read_csv('finals.csv', index_col='sno')
temp = df['Temp']

def dt_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        x.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(x), np.array(y)

WINDOW_SIZE = 12
x, y = dt_to_X_y(temp, window_size=WINDOW_SIZE)
#print(X.shape, y.shape)
#print(X[:5])
#print(y[:5])
#print(len(X)*.8, len(y)*.1)
x_train, y_train = x[:int(len(x)*0.8)], y[:int(len(y)*0.8)]
x_test, y_test = x[int(len(x)*0.8):], y[int(len(y)*0.8):]

# model = Sequential()
# model.add(InputLayer(input_shape=(WINDOW_SIZE, 1)))
# model.add(LSTM(64))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))

model = Sequential()
model.add(InputLayer(input_shape=(WINDOW_SIZE, 1)))
model.add(Conv1D(64, kernel_size=3))
model.add(Flatten()) 
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=["accuracy"])



# Define Flower client
class CifarClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(x_train, y_train, epochs=10, batch_size=32)


    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(loss, accuracy)

    print(type(loss), type(accuracy))
    #plt.plot(loss, accuracy)
    #plt.show()
    
   
    # # Calculate RMSE
    # rmse = np.sqrt(np.mean((y_test - model.predict(x_test)) ** 2))
    # print("RMSE: ", rmse)
    # # Calculate MAE 
    # mae = np.mean(np.abs(y_test - model.predict(x_test)))
    # print("MAE: ", mae)
    # # Calculate MAPE
    # mape = np.mean(np.abs((y_test - model.predict(x_test)) / y_test)) * 100
    # print("MAPE: ", mape)

    return loss, len(x_test), {"accuracy": accuracy} 

# Start Flower client
fl.client.start_numpy_client(server_address="192.168.10.103:8080", client=CifarClient())


