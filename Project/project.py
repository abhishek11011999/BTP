import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model

# Preprocessing of data
# Reading of data

data = pd.read_excel (r'/content/gdrive/My Drive/VideoTestData_23.1.16-29.1.16.xlsx') 
df = pd.DataFrame(data)
df = df.iloc[1:]
df.to_csv('/content/gdrive/My Drive/VideoTestData_23.1.16-29.1.16.csv', header=False, index=False)

# create DataFrame

ts = pd.Timestamp
df = pd.read_csv('/content/gdrive/My Drive/VideoTestData_23.1.16-29.1.16.csv')
df = df[['ID','Test Time','Celluar Signal Strength(dBm)','Video Total DL Rate(kbps)']]
df['Test Time'] = pd.to_datetime(df['Test Time'])

# Binning of data into various time slot

df2 = pd.DataFrame(columns=['Test Time','Celluar Signal Strength(dBm) mean','Video Total DL Rate(kbps) mean'])
index = pd.date_range('20160123',periods=7*(24*60)+1,freq='T') # for the gap of 1 min. bin
for i in range(7*(24*60)+1):
  try:
    start_time = index[0+i]
    end_time = index[1+i]
    mask = (df['Test Time'] > start_time) & (df['Test Time'] <= end_time)
    temp = df.loc[mask]
    df2.loc[i] = [start_time] + [temp['Celluar Signal Strength(dBm)'].mean()] + [temp['Video Total DL Rate(kbps)'].mean()]
    print(df2[['Celluar Signal Strength(dBm)','Video Total DL Rate(kbps)']].mean())
  except:
    continue

# Plotting of DataFrame to see the trend of data

# for normal plot

#ax = plt.gca()
#df2.plot(kind='line',x='Test Time',y='Celluar Signal Strength(dBm) mean',ax=ax)
#df2.plot(kind='line',x='Test Time',y='Video Total DL Rate(kbps) mean', color='red')
#plt.show()

# for sacttered plot

sns.lmplot( x="Test Time", y="Celluar Signal Strength(dBm) mean", data=df2, fit_reg=False, legend=False, height=8, aspect=1.5, scatter_kws={"s": 10,"color":"red"})
sns.lmplot( x="Test Time", y="Video Total DL Rate(kbps) mean", data=df2, fit_reg=False, legend=False, height=8, aspect=1.5, scatter_kws={"s": 10,"color":"red"})

# Filling of NaN value by its previous value and cliping

df3 = df2[['Test Time','Celluar Signal Strength(dBm) mean','Video Total DL Rate(kbps) mean']]
df3['signal_strength(dBm)'] = df3['Celluar Signal Strength(dBm) mean'].fillna(method='ffill').fillna(method='bfill')
df3['Video_total_dl_rate(kbps)'] = df3['Video Total DL Rate(kbps) mean'].fillna(method='ffill').fillna(method='bfill')
df3 = df3.drop(columns=['Celluar Signal Strength(dBm) mean','Video Total DL Rate(kbps) mean'])
df3['signal_strength(dBm)'] = df3['signal_strength(dBm)'].clip(-110,-70)
df3['Video_total_dl_rate(kbps)'] = df3['Video_total_dl_rate(kbps)'].clip(1200,2400) # define range so that sudden spike in data gets removed

#ax = plt.gca()
#df3.plot(kind='line',x='Test Time',y='signal_strength(dBm)',ax=ax)
#df3.plot(kind='line',x='Test Time',y='Video_total_dl_rate(kbps)', color='r')

sns.lmplot( x="Test Time", y="signal_strength(dBm)", data=df3, fit_reg=False, legend=False, height=8, aspect=1.5, scatter_kws={"s": 10,"color":"red"})
sns.lmplot( x="Test Time", y="Video_total_dl_rate(kbps)", data=df3, fit_reg=False, legend=False, height=8, aspect=1.5, scatter_kws={"s": 10,"color":"red"})

# Prediction using RNN LSTM
# Preprocessing of data before training.
train_set = df3.iloc[:, 2:3].values

sc = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = sc.fit_transform(train_set)

X_train = []
y_train = []

for i in range(60, 8638):
    X_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Training phase of LSTM model.
# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 60)

# Saving and Loading of model.

regressor.save("/content/gdrive/My Drive/project/regressor.h5")
regressor = load_model("/content/gdrive/My Drive/project/regressor.h5")

# Various types of Error/Accuracy measured in LSTM model

data_set = df3.iloc[:, 2:3].values
realx_test_data = data_set[8640:]

test_set = data_set[10080-1440-60:]
test_set = test_set.reshape(-1,1)
test_set = sc.transform(test_set)

X_test = []
for i in range(60, 1500):
    X_test.append(test_set[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_x = regressor.predict(X_test)
predicted_x = sc.inverse_transform(predicted_x)

sum_n = 0

for i in range(1440):
  sum_n = sum_n + abs(realx_test_data[i]-predicted_x[i])

avg = sum_n/1440
print("Mean absolute error(MAE):",avg)

sum_p = 0

for i in range(1440):
  sum_p = sum_p + pow(realx_test_data[i]-predicted_x[i],2)

avg = sum_p/1440
avg_s = math.sqrt(avg)
print("Root mean square error(RMSE):",avg_s)

sum_a = 0

for i in range(1440):
  sum_a = sum_a + (abs(realx_test_data[i]-predicted_x[i])/abs(realx_test_data[i]))

avg_a = (sum_a/1440)*100
print("Mean absolute percentage error(MASE):",avg_a)

# plot of real data vs predicted data

df['realx_test_data'] = pd.DataFrame(data_set[8640:])
df['Video_dl_speed(kbps)'] = pd.DataFrame(predicted_x)
df['Time Bin(gap of 1 min.)'] = pd.DataFrame([int(x) for x in range(1440)])

# normal plot

# plt.figure(figsize=(6.4,6.0),dpi=100)
# plt.plot(realx_test_data, color = 'red', label = 'Real data')
# plt.plot(predicted_x, color = 'blue', label = 'Predicted data')
# plt.title('Video Total Download Speed')
# plt.xlabel('Time_bin(Gap of 1 min.)')
# plt.ylabel('Download_speed(kbps)')
# plt.legend()
# plt.show()

# scattered plot

fig, axs = plt.subplots(ncols=1)
fig.set_size_inches(16.5, 8.5)
sns.regplot(x='Time Bin(gap of 1 min.)', y='realx_test_data', data=df, fit_reg = False, scatter_kws={"s": 50})
sns.regplot(x='Time Bin(gap of 1 min.)', y='Video_dl_speed(kbps)', data=df, marker="+", fit_reg = False, scatter_kws={"s": 50})


# Prediction using linear regression
# Linear Regression model,Error/Accuracy and plot of real vs predicted data

arr = [[i] for i in range(10080)]
x = np.array(arr)
y = df3['Video_total_dl_rate(kbps)'].to_numpy()
print(type(x),type(y))

x_train = x[0:8640]
x_test = x[8640:]

y_train = y[0:8640]
y_test = y[8640:]

regression_model = LinearRegression()

# Fit the data(train the model)

regression_model.fit(x_train, y_train)

# Predict

y_predicted = regression_model.predict(x_test)

# model evaluation

rmse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

# printing values

print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# plotting values

# data points
# plt.plot(x,y)
# plt.show()

plt.figure(figsize=(6.4,6.0),dpi=100)
plt.plot(x_test,y_test)
plt.plot(x_test,y_predicted,color='r')
plt.title('Video Total Download Speed')
plt.xlabel('Time_bin(Gap of 1 min.)')
plt.ylabel('Download_speed(kbps)')
plt.legend()