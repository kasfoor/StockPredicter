import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

#connect to Quandl API
quandl.ApiConfig.api_key = '4t1rpFDecswedxkyqyFS'

#Pick which stock to analyze and set df to Adj. Close
df = quandl.get("WIKI/AMZN")
df = df[["Adj. Close"]]

#Plot stock based on current data
df['Adj. Close'].plot(figsize = (15,6),color='g')
plt.legend(loc = 'upper left')
plt.show()

#Pick forecast for prediction and make df for predicted range
forecast = 30
df['Prediction'] = df[['Adj. Close']].shift(-forecast)

#preprocess data to set x mean to 0 and standard deviation to 1
x = np.array(df.drop(['Prediction'],1))
x = preprocessing.scale(x)

#Adjust our x and y for forecast range
x_forecast = x[-forecast:]
x = x[:-forecast]
y = np.array(df['Prediction'])
y = y[:-forecast]

#Use training split with values and set training size to 80% of data and testing to 20%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
clf = LinearRegression()
clf.fit(x_train,y_train)

#Calcuate the confidence using linear regression
confidence = clf.score(x_test, y_test)
forecast_predicted = clf.predict(x_forecast)

#Change date range and plot the graph
dates = pd.date_range(start = "2018-03-28", end = "2018-04-26")
plt.plot(dates,forecast_predicted,color="y")
df['Adj. Close'].plot(color='g')
plt.xlim(xmin=datetime.date(2017,4,26),xmax=datetime.date(2018,5,1))
plt.show()