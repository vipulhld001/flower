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
from tensorflow.keras.metrics import Accuracy

#df = pd.read_csv('AllWeather.csv', index_col='sno')
df = pd.read_csv('finals.csv', index_col='sno')
##To plot sample data
#print(df.head())
#df.plot(figsize=(10, 5))
#####
#print(df[:25])


#Plotting Temp Values
temp = df['Temp']
temp.plot()
#plt.show()
#######

#[[[1],[2],[3],[4],[5]]] [6] we give 1 to 5 hours and we get 6th hour value
# [[2,3,4,5,6]] [7] we give 2 to 6 hours and we get 7th hour value
# [[3,4,5,6,7]] [8] we give 3 to 7 hours and we get 8th hour value 
#We predict the next value based on the previous 5 values

#Supervoised learning

def dt_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = dt_to_X_y(temp, window_size=WINDOW_SIZE)
print(X.shape, y.shape)
#print(X[:5])
#print(y[:5])
print(len(X)*.8, len(y)*.1)
X_train, y_train = X[:int(len(X)*0.8)], y[:int(len(y)*0.8)]
X_val, y_val = X[int(len(X)*0.8):int(len(X)*0.9)], y[int(len(y)*0.8):int(len(y)*0.9)]
X_test, y_test = X[int(len(X)*0.9):], y[int(len(y)*0.9):]

#Print all shapes
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape) 
print(X_test.shape, y_test.shape)
#All  shape printed



###MOdel

model1 = Sequential()
model1.add(InputLayer(input_shape=(WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1, activation='linear'))

model1.summary()

#Model Cleckpoint

cp = ModelCheckpoint('model1/', save_best_only=True)

model1.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=["accuracy"])

history = model1.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), callbacks=[cp])

score = model1.evaluate(X_test, y_test)
print("Test Score= ", score[0])
print("Test Accuracy= ", score[1])
##Loading the model

model1 = load_model('model1/')

train_prdictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Prediction': train_prdictions, 'Actuals': y_train})
print(train_results)

plt.plot(train_results['Train Prediction'][:10])
plt.plot(train_results['Actuals'][:10]) 
plt.show()