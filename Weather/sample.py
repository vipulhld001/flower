import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df = pd.read_csv('AllWeather.csv', index_col='sno')
df = pd.read_csv('finals.csv', index_col='sno')
##To plot sample data
print(df.head())
#df.plot(figsize=(10, 5))
#####
print(df[:25])

temp = df['Temp']
temp.plot()
plt.show()

