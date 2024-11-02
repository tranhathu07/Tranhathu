import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('advertising1.csv',delimiter = ',', skip_header = 1)

N= data.shape[0]
X = data[:,:3]
y = data[:,3:]

def normalization(x):
    N = len(x)
    maxi = np.max(x)
    mini = np.min(x)
    aver = np.mean(x)
    x = (x-aver)/(maxi - mini)
    x_b = np.c_[np.ones((N,1)),x]
    return x_b,maxi,mini,aver

def stochastic_gradient_descent(x_b,y,n_epochs=50,learning_rate=0.00001):
    # thetas = np.random.randn(4,1)
    thetas = np.array([[1.16270837] , [ -0.81960489] , [1.39501033] ,[0.29763545]])
    thetas_path = [thetas]
    losses = []
    for epoch in range(n_epochs):
        for i in range(N):
            #random_index = np.random.randint(N)
            random_index = i

            xi = x_b[random_index:random_index +1]
            yi = y[random_index:random_index+1]

        #Compute output
            y_hat = xi.dot(thetas)

            #Comput loss
            loss = ((yi - y_hat)**2)/2

            #Compute gradient:
            gradient = xi.T.dot(y_hat -yi)

            #Compute theta:
            thetas = thetas - learning_rate*gradient
            thetas_path.append(thetas.copy())
            losses.append(loss.item())

    return thetas_path,losses
  
x_b,maxi,mini,aver = normalization(X)
sgd_theta, losses = stochastic_gradient_descent(x_b,y,n_epochs = 1, learning_rate = 0.01)

def mini_batch_gradient_descent(x_b,y,n_epochs = 50,minibatch_size = 20,learning_rate = 0.01):
    N = x_b.shape[0]
    thetas = np.asarray ([[1.16270837] , [-0.81960489] , [1.39501033] ,[0.29763545]])
    thetas_path= [thetas]
    losses = []
    for epoch in range(n_epochs):
        #shuffle_indices = np.random.permutation(N)
        shuffled_indices = np.asarray ([21 , 144 , 17 , 107 , 37 , 115 , 167 , 31 , 3 ,
132 , 179 , 155 , 36 , 191 , 182 , 170 , 27 , 35 , 162 , 25 , 28 , 73 , 172 , 152 , 102 , 16 ,
185 , 11 , 1 , 34 , 177 , 29 , 96 , 22 , 76 , 196 , 6 , 128 , 114 , 117 , 111 , 43 , 57 , 126 ,
165 , 78 , 151 , 104 , 110 , 53 , 181 , 113 , 173 , 75 , 23 , 161 , 85 , 94 , 18 , 148 , 190 ,
169 , 149 , 79 , 138 , 20 , 108 , 137 , 93 , 192 , 198 , 153 , 4 , 45 , 164 , 26 , 8 , 131 ,
77 , 80 , 130 , 127 , 125 , 61 , 10 , 175 , 143 , 87 , 33 , 50 , 54 , 97 , 9 , 84 , 188 , 139 ,
195 , 72 , 64 , 194 , 44 , 109 , 112 , 60 , 86 , 90 , 140 , 171 , 59 , 199 , 105 , 41 , 147 ,
92 , 52 , 124 , 71 , 197 , 163 , 98 , 189 , 103 , 51 , 39 , 180 , 74 , 145 , 118 , 38 , 47 ,
174 , 100 , 184 , 183 , 160 , 69 , 91 , 82 , 42 , 89 , 81 , 186 , 136 , 63 , 157 , 46 , 67 ,
129 , 120 , 116 , 32 , 19 , 187 , 70 , 141 , 146 , 15 , 58 , 119 , 12 , 95 , 0 , 40 , 83 , 24 ,
168 , 150 , 178 , 49 , 159 , 7 , 193 , 48 , 30 , 14 , 121 , 5 , 142 , 65 , 176 , 101 , 55 ,
133 , 13 , 106 , 66 , 99 , 68 , 135 , 158 , 88 , 62 , 166 , 156 , 2 , 134 , 56 , 123 , 122 ,
154])
        x_b_shuffled = x_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0,N,minibatch_size):
            xi = x_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]

            #Compute output
            y_hat = xi.dot(thetas)

            #compute loss
            loss_mean = (1/2)*np.mean((yi-y_hat)**2)

            #Compute gradient
            gradient = xi.T.dot(y_hat - yi) / minibatch_size

            #Compute thetas
            thetas = thetas - learning_rate*gradient
            thetas_path.append(thetas)
            losses.append(loss_mean)
    return thetas_path,losses

# sgd_thetas,losses = mini_batch_gradient_descent(x_b,y,n_epochs = 50,minibatch_size = 20,learning_rate = 0.01)

# x_axis = list(range(200))
# plt.plot(x_axis,losses[:200],color = 'r')
# plt.show()

# mbgd_thetas,losses = mini_batch_gradient_descent(x_b,y,n_epochs = 50,minibatch_size = 20,learning_rate = 0.01)
# print(round(sum(losses)),2)

def batch_gradient_descent(x_b,y,n_epochs = 100,learning_rate = 0.01 ):
    # thetas = np. random . randn (4 , 1) # uncomment this line for real application
    thetas = np.asarray ([[1.16270837] , [ -0.81960489] , [1.39501033] ,
[0.29763545]])
    thetas_path = [thetas]
    losses = []
    N = x_b.shape[0]
    for i in range(n_epochs):
        #Compute_output
        y_hat = x_b.dot(thetas)
        loss = (1/2) *(np.mean((y-y_hat)**2))
        gradient = (1/N) *(x_b.T.dot(y_hat-y))
        thetas = thetas - gradient*learning_rate
        thetas_path.append(thetas)
        mean_loss = (np.sum(loss))/N
        losses.append(mean_loss)
    return thetas_path,losses

bgd_thetas,losses = batch_gradient_descent(x_b,y,n_epochs = 100,learning_rate = 0.01)
# print(round(sum(losses),2))

######## Bài 2 #######
import pandas as pd
df = pd.read_csv('BTC-Daily.csv')
df = df.drop_duplicates()

df['date'] = pd.to_datetime(df['date'])
date_range = str(df['date'].dt.date.min()) + ' to ' + str(df['date'].dt.date.max())
print(date_range)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

unique_years = df['year'].unique()
for year in unique_years:

    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    year_month_day = pd.DataFrame({'date': dates})
    year_month_day['year'] = year_month_day['date'].dt.year
    year_month_day['month'] = year_month_day['date'].dt.month
    year_month_day['day'] = year_month_day['date'].dt.day


    merged_data = pd.merge(year_month_day, df, on=['year', 'month', 'day'], how='left')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['date_x'], merged_data['close'])
    plt.title(f'Bitcoin Closing Prices - {year}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#!pip install mplfinance

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import datetime

# Filter data for 2019-2022
df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2022-12-31')]

# Convert date to matplotlib format
df_filtered['date'] = df_filtered['date'].map(mdates.date2num)

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(20, 6))

candlestick_ohlc(ax, df_filtered[['date', 'open', 'high', 'low', 'close']].values, width=0.6, colorup='g', colordown='r')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.title('Bitcoin Candlestick Chart (2019-2022)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)

# Save the plot as a PDF
plt.savefig('bitcoin_candlestick_2019_2022.pdf')

plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae
scalar = StandardScaler()

df["Standardized_Close_Prices"] = scalar.fit_transform(df["close"].values.reshape(-1,1))
df["Standardized_Open_Prices"] = scalar.fit_transform(df["open"].values.reshape(-1,1))
df["Standardized_High_Prices"] = scalar.fit_transform(df["high"].values.reshape(-1,1))
df["Standardized_Low_Prices"] = scalar.fit_transform(df["low"].values.reshape(-1,1))

#Converting Date to numerical form

df['date_str'] = df['date'].dt.strftime('%Y%m%d%H%M%S')

# Convert the string date to a numerical value
df['NumericalDate'] = pd.to_numeric(df['date_str'])

# Drop the intermediate 'date_str' column if not needed
df.drop(columns=['date_str'], inplace=True)


X = df[["Standardized_Open_Prices", "Standardized_High_Prices", "Standardized_Low_Prices"]]
y = df["Standardized_Close_Prices"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

b = 0
w = np.zeros(X_train.shape[1])
lr = 0.01
epochs = 200

def predict(X,w,b):
    return X.dot(w) + b
def gradient(y_hat,y,x):
    loss = y_hat - y
    dw = x.T.dot(loss)/len(y)
    db = np.sum(loss)/len(y)
    cost = np.sum(loss**2)/(2*len(y))
    return (dw,db,cost)
def update_weight(w,b,dw,db,lr):
    w_new = w - lr*dw
    b_new = b - lr*db
    return (w_new,b_new)

def linear_regression_vectorized(X,y,lr = 0.01,num_iteration =200):
    n_samples,n_feature = X.shape
    w = np.zeros(n_feature)
    b = 0
    losses= []
    for _ in range(num_iteration):
        y_hat = predict(X,w,b)
        dw,db,cost = gradient(y_hat,y,X)
        w,b = update_weight(w,b,dw,db,lr)
        losses.append(cost)
    return w,b,losses
w, b, losses = linear_regression_vectorized(X_train.values, y_train.values, lr=0.01, num_iteration=200)
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()

from sklearn.metrics import r2_score

# Make predictions on the test set
y_pred = predict(X_test, w, b)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

# Calculate MAE
mae = np.mean(np.abs(y_pred - y_test))

# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


# Calculate R-squared on training data
y_train_pred = predict(X_train, w, b)
train_accuracy = r2_score(y_train, y_train_pred)

# Calculate R-squared on testing data
test_accuracy = r2_score(y_test, y_pred)

print("Root Mean Square Error (RMSE):", round(rmse, 4))
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Training Accuracy (R-squared):", round(train_accuracy, 4))
print("Testing Accuracy (R-squared):", round(test_accuracy, 4))

# Filter data for 2019-01-01 to 2019-04-01
df_2019_Q1 = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2019-04-01')]

# Prepare X and y for prediction
X_2019_Q1 = df_2019_Q1[["open", "high", "low"]]
y_2019_Q1_actual = df_2019_Q1["close"]

# Predict using the trained model
y_2019_Q1_pred = predict(X_2019_Q1, w, b)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df_2019_Q1['date'], y_2019_Q1_actual, label='Actual Close Price', marker='o')
plt.plot(df_2019_Q1['date'], y_2019_Q1_pred, label='Predicted Close Price', marker='x')
plt.title('Actual vs. Predicted Bitcoin Close Price (01/01/2019 - 04/01/2019)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filter data for 2020
df_2020 = df[(df['date'] >= '2020-01-01') & (df['date'] <= '2020-12-31')]

# Create a new column for quarter
df_2020['quarter'] = df_2020['date'].dt.quarter

# Prepare X and y for prediction
X_2020 = df_2020[["open", "high", "low"]]
y_2020 = df_2020["close"]

# Make predictions for 2020
y_pred_2020 = predict(X_2020, w, b)


# Plot actual vs. predicted close prices for each quarter
for quarter in df_2020['quarter'].unique():
  df_quarter = df_2020[df_2020['quarter'] == quarter]
  plt.figure(figsize=(10, 6))
  plt.plot(df_quarter['date'], df_quarter['close'], label='Actual')
  plt.plot(df_quarter['date'], y_pred_2020[df_2020['quarter'] == quarter], label='Predicted')
  plt.title(f'Actual vs. Predicted Bitcoin Close Price (2020) - Quarter {quarter}')
  plt.xlabel('Date')
  plt.ylabel('Close Price (USD)')
  plt.legend()
  plt.grid(True)
  plt.show()

