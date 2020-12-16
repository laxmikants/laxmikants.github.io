---
title: "Applying Logistic Regression"
author: Laxmi K Soni 
description: "A case study on applying logistic regression to stock market"
slug: Applying Logistic Regression
date: 2020-03-26
lastmod: 2020-03-26
categories: ["Stock Market","Logistic Regression"]
tags: ["Stock Market","Logistic Regression"]
Summary: Application of Logistic Regression to Stock movement prediction
subtitle: Application of Logistic Regression to Stock movement prediction
featured: "img/main/logistic-stock-8.jpg"
output:
  blogdown::html_page:
    toc: true
    number_sections: true
    keep_md: true
  html_document:
    highlight: tango
    theme: flatly
    toc: no
    keep_md: no
---




***1:Loading Stock data***

***1.1:Importing libraries and data***


```python
import investpy
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from scipy import stats 
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import talib
import quandl
```


***1.2:Fetching the data***

To fetch the stock data we use `investpy` library. This library fetchs data from investing for example:


```python
df = investpy.commodities.get_commodity_historical_data(stock='ABC',from_date='01/01/2015',to_date='14/08/2020', country = "India")
```




```python

```



```
##               Open    High     Low   Close  Volume
## Date                                              
## 2019-12-02  44.290  44.416  44.077  44.344     354
## 2019-12-03  44.250  44.985  44.150  44.620     138
## 2019-12-04  44.949  45.100  44.100  44.392      34
## 2019-12-05  44.200  45.287  43.850  44.064      44
## 2019-12-06  44.599  44.714  43.470  43.545   24469
## ...            ...     ...     ...     ...     ...
## 2020-12-07  63.643  65.650  62.380  65.499   26874
## 2020-12-08  65.388  65.817  64.576  65.192   20209
## 2020-12-09  64.521  64.804  63.171  63.499   26603
## 2020-12-10  63.747  64.399  62.931  63.530   19637
## 2020-12-11  63.459  63.935  62.700  63.735   19691
## 
## [266 rows x 5 columns]
```

```
## Date
## 2019-12-31    44.881810
## 2020-01-31    46.695000
## 2020-02-29    46.520429
## 2020-03-31    41.378909
## 2020-04-30    42.363278
## 2020-05-31    45.519857
## 2020-06-30    48.408409
## 2020-07-31    55.729304
## 2020-08-31    68.659667
## 2020-09-30    65.037545
## 2020-10-31    61.676524
## 2020-11-30    62.029773
## 2020-12-31    63.408778
## Freq: M, Name: Close, dtype: float64
```

```
##               Open    High     Low   Close  Volume Currency  usdinr  usdsilver
## Date                                                                          
## 2019-12-02  44.290  44.416  44.077  44.344     354      INR  71.634     17.141
## 2019-12-03  44.250  44.985  44.150  44.620     138      INR  71.740     17.422
## 2019-12-04  44.949  45.100  44.100  44.392      34      INR  71.474     17.088
## 2019-12-05  44.200  45.287  43.850  44.064      44      INR  71.252     17.227
## 2019-12-06  44.599  44.714  43.470  43.545   24469      INR  71.283     16.763
## ...            ...     ...     ...     ...     ...      ...     ...        ...
## 2020-12-07  63.643  65.650  62.380  65.499   26874      INR  73.826     24.794
## 2020-12-08  65.388  65.817  64.576  65.192   20209      INR  73.720     24.736
## 2020-12-09  64.521  64.804  63.171  63.499   26603      INR  73.720     23.990
## 2020-12-10  63.747  64.399  62.931  63.530   19637      INR  73.740     24.094
## 2020-12-11  63.459  63.935  62.700  63.735   19691      INR  73.736     24.092
## 
## [266 rows x 8 columns]
```

## z-scores


```python

df['zscore'] = stats.zscore(df['Close'])

print(df['zscore'].tail(30))
```

```
## Date
## 2020-11-03    1.012342
## 2020-11-04    0.878218
## 2020-11-05    1.174615
## 2020-11-06    1.286592
## 2020-11-09    0.822851
## 2020-11-10    1.049495
## 2020-11-11    0.997439
## 2020-11-12    1.017930
## 2020-11-13    1.120800
## 2020-11-14    1.107864
## 2020-11-16    1.116453
## 2020-11-17    1.070607
## 2020-11-18    0.997646
## 2020-11-19    0.890741
## 2020-11-20    0.957802
## 2020-11-23    0.788802
## 2020-11-24    0.695247
## 2020-11-25    0.718222
## 2020-11-26    0.721327
## 2020-11-27    0.768932
## 2020-11-30    0.643605
## 2020-12-01    0.932965
## 2020-12-02    0.957285
## 2020-12-03    1.012031
## 2020-12-04    0.990195
## 2020-12-07    1.303564
## 2020-12-08    1.271793
## 2020-12-09    1.096583
## 2020-12-10    1.099791
## 2020-12-11    1.121007
## Name: zscore, dtype: float64
```

```python
pd.set_option('display.max_columns', 10)
PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
R1 = pd.Series(2 * PP - df['Low'])  
S1 = pd.Series(2 * PP - df['High'])  
R2 = pd.Series(PP + df['High'] - df['Low'])  
S2 = pd.Series(PP - df['High'] + df['Low'])  
R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
PSR = pd.DataFrame(psr)  
print(PSR.tail())
```

```
##                    PP         R1         S1         R2         S2         R3  \
## Date                                                                           
## 2020-12-07  64.509667  66.639333  63.369333  67.779667  61.239667  69.909333   
## 2020-12-08  65.195000  65.814000  64.573000  66.436000  63.954000  67.055000   
## 2020-12-09  63.824667  64.478333  62.845333  65.457667  62.191667  66.111333   
## 2020-12-10  63.620000  64.309000  62.841000  65.088000  62.152000  65.777000   
## 2020-12-11  63.456667  64.213333  62.978333  64.691667  62.221667  65.448333   
## 
##                    S3  
## Date                   
## 2020-12-07  60.099333  
## 2020-12-08  63.332000  
## 2020-12-09  61.212333  
## 2020-12-10  61.373000  
## 2020-12-11  61.743333
```

As we can see, we now have a data frame with all the entries from start date to end date. We have multiple columns here and not only the closing stock price of the respective day. Let’s take a quick look at the individual columns and their meaning.

`Open:` That’s the share price the stock had when the markets opened that day.

`Close:` That’s the share price the stock had when the markets closed that day.

`High:` That’s the highest share price that the stock had that day.

`Low:` That’s the lowest share price that the stock had that day.

`Volume:` Amount of shares that changed hands that day.

***1.3:Reading individual values***

Since our data is stored in a Pandas data frame, we can use the indexing we already know, to get individual values. For example, we could only print the closing values using `print (df[ 'Close' ])`

Also, we can go ahead and print the closing value of a specific date that we are interested in. This is possible because the date is our index column.


```python
print (df[ 'Close' ][ '2020-07-14' ])
```

```
## 52.649
```


But we could also use simple indexing to access certain positions.


```python
print (df[ 'Close' ][ 5 ])
```

```
## 43.502
```

Here we  printed the closing price of the fifth entry.


***2:Graphical Visualization***

Even though tables are nice and useful, we want to visualize our financial data, in order to get a better overview. We want to look at the development of the share price. 

Actually plotting our share price curve with Pandas and Matplotlib is very simple. Since Pandas builds on top of Matplotlib, we can just select the column we are interested in and apply the plot method. The results are amazing. Since the date is the index of our data frame, Matplotlib uses it for the x-axis. The y-values are then our adjusted close values.



```python
import matplotlib.pyplot as plt
from matplotlib import style
style.use( 'ggplot' )
plt.ylabel( 'Close' )
plt.title( 'Share Price' )
df['Close'].plot()
plt.show()
```

![](/img/main/stockplot.png)

***2.1:CandleStick Charts***

The best way to visualize stock data is to use so-called candlestick charts . This type of chart gives us information about four different values at the same time, namely the high, the low, the open and the close value. In order to plot candlestick charts, we will need to import a function of the MPL-Finance library. 


```python
import mplfinance as fplt
```
We are importing the candlestick_ohlc function. Notice that there also exists a candlestick_ochl function that takes in the data in a different order.
Also, for our candlestick chart, we will need a different date format provided by Matplotlib. Therefore, we need to import the respective module as well. We give it the alias mdates .


```python
import matplotlib.dates as mdates
```

***2.2: Preparing the data for CandleStick charts***

Now in order to plot our stock data, we need to select the four columns in the right order.


```python
df1 = df[[ 'Open' , 'High' , 'Low' , 'Close' ]]
```

Now, we have our columns in the right order but there is still a problem. Our date doesn’t have the right format and since it is the index, we cannot manipulate it. 
Therefore, we need to reset the index and then convert our datetime to a number.


```python
df1.reset_index( inplace = True )
df1[ 'Date' ] = df1[ 'Date' ].map(mdates.date2num)
```

```
## C:/Users/slaxm/AppData/Local/r-miniconda/envs/r-reticulate/python.exe:1: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```
For this, we use the reset_index function so that we can manipulate our Date column. Notice that we are using the inplace parameter to replace the data frame by the new one. After that, we map the date2num function of the matplotlib.dates module on all of our values. That converts our dates into numbers that we can work with.

***2.3:Plotting the data***

Now we can start plotting our graph. For this, we just define a subplot (because we need to pass one to our function) and call our candlestick_ohlc function.


```python
ax = plt.subplot()
fplt(ax, df1.values,
                  width = 5 ,
                  colordown = 'r' , colorup = 'g' )
ax.grid()
ax.xaxis_date()
plt.show()
```

![](/img/main/candlestick.png)

One candlestick gives us the information about all four values of one specific day. The highest point of the stick is the high and the lowest point is the low of that day. The colored area is the difference between the open and the close  price. 
If the stick is green, the close value is at the top and the open value at the bottom, since the close must be higher than the open. If it is red, it is the other way around.

***2.4:Analysis and Statistics ***

Now let’s get a little bit deeper into the numbers here and away from the visual. 
From our data we can derive some statistical values that will help us to analyze it.



***`PERCENTAGE CHANGE`***

One value that we can calculate is the percentage change of that day. This means by how many percent the share price increased or decreased that day.


```python
df[ 'PCT_Change' ] = (df[ 'Close' ] - df[ 'Open' ]) / df[ 'Open' ]
```

The calculation is quite simple. We create a new column with the name PCT_Change and the values are just the difference of the closing and opening values divided by the opening values. Since the open value is the beginning value of that day, we take it as a basis. We could also multiply the result by 100 to get the actual percentage.

*** `HIGH LOW PERCENTAGE` ***

Another interesting statistic is the high low percentage. Here we just calculate the difference between the highest and the lowest value and divide it by the closing value.

By doing that we can get a feeling of how volatile the stock is.


```python
df[ 'HL_PCT' ] = (df[ 'High' ] - df[ 'Low' ]) / df[ 'Close' ]
print(df.tail())
```

```
##               Open    High     Low   Close  Volume  ...  usdinr  usdsilver  \
## Date                                                ...                      
## 2020-12-07  63.643  65.650  62.380  65.499   26874  ...  73.826     24.794   
## 2020-12-08  65.388  65.817  64.576  65.192   20209  ...  73.720     24.736   
## 2020-12-09  64.521  64.804  63.171  63.499   26603  ...  73.720     23.990   
## 2020-12-10  63.747  64.399  62.931  63.530   19637  ...  73.740     24.094   
## 2020-12-11  63.459  63.935  62.700  63.735   19691  ...  73.736     24.092   
## 
##               zscore  PCT_Change    HL_PCT  
## Date                                        
## 2020-12-07  1.303564    0.029163  0.049924  
## 2020-12-08  1.271793   -0.002997  0.019036  
## 2020-12-09  1.096583   -0.015840  0.025717  
## 2020-12-10  1.099791   -0.003404  0.023107  
## 2020-12-11  1.121007    0.004349  0.019377  
## 
## [5 rows x 11 columns]
```
These statistical values can be used with many others to get a lot of valuable information about specific stocks. This improves the decision making


*** `MOVING AVERAGE` ***

we are going to derive the different moving averages . It is the arithmetic mean of all the values of the past n days. Of course this is not the only key statistic that we can derive, but it is the one we are going to use now. We can play around with other functions as well.

What we are going to do with this value is to include it into our data frame and to compare it with the share price of that day.

For this, we will first need to create a new column. Pandas does this automatically when we assign values to a column name. This means that we don’t have to explicitly define that we are creating a new column. 


```python

df[ '5d_ma' ] = round(df[ 'Close' ].rolling( window = 5 , min_periods = 0 ).mean(),2)
df[ '20d_ma' ] = round(df[ 'Close' ].rolling( window = 20 , min_periods = 0 ).mean(),2)
df['50d_ma'] = round(df[ 'Close' ].rolling(window = 50, min_periods = 0).mean(),2)
df['100d_ma'] = round(df[ 'Close' ].rolling(window = 100, min_periods = 0).mean(),2)
df['200d_ma'] = round(df[ 'Close' ].rolling(window = 200, min_periods = 0).mean(),2)

#df[ '20d_ma' ] = round(df.Close.ewm(span=21, adjust=False).mean(),2)
#df['50d_ma'] = round(df.Close.ewm(span=49, adjust=False).mean(),2)

#df['100d_ma'] = round(df.Close.ewm(span=98, adjust=False).mean(),2)
#df['200d_ma'] = round(df.Close.ewm(span=196, adjust=False).mean(),2)

print(df[['Close','5d_ma','20d_ma','50d_ma','100d_ma','200d_ma']].tail(7))
```

```
##              Close  5d_ma  20d_ma  50d_ma  100d_ma  200d_ma
## Date                                                       
## 2020-12-03  62.682  61.24   61.79   61.77    63.91    54.64
## 2020-12-04  62.471  61.67   61.87   61.84    64.00    54.71
## 2020-12-07  65.499  62.94   61.99   61.94    64.12    54.80
## 2020-12-08  65.192  63.60   62.12   61.99    64.20    54.89
## 2020-12-09  63.499  63.87   62.16   62.07    64.22    54.98
## 2020-12-10  63.530  64.04   62.15   62.11    64.24    55.08
## 2020-12-11  63.735  64.29   62.16   62.15    64.27    55.17
```

Here we define a three new columns with the name 20d_ma, 50d_ma, 100d_ma,200d_ma . We now fill this column with the mean values of every n entries. The rolling function stacks a specific amount of entries, in order to make a statistical calculation possible. The window parameter is the one which defines how many entries we are going to stack. But there is also the min_periods parameter. This one defines how many entries we need to have as a minimum in order to perform the calculation. This is relevant because the first entries of our data frame won’t have a n entries previous to them. By setting this value to zero we start the calculations already with the first number, even if there is not a single previous value. This has the effect that the first value will be just the first number, the second one will be the mean of the first two numbers and so on, until we get to a b values.

By using the mean function, we are obviously calculating the arithmetic mean. However, we can use a bunch of other functions like max, min or median if we like to.



*** `Standard Deviation` ***

The variability of the closing stock prices determinies how vo widely prices are dispersed from the average price. If the prices are trading in narrow trading range the standard deviation will return a low value that indicates low volatility. If the prices are trading in wide trading range the standard deviation will return high value that indicates high volatility.


```python
df['Std_dev']= df['Close'].rolling(7).std()  
print(df['Std_dev'].tail(7))
```

```
## Date
## 2020-12-03    1.379397
## 2020-12-04    1.417915
## 2020-12-07    2.000723
## 2020-12-08    2.152803
## 2020-12-09    1.456908
## 2020-12-10    1.314193
## 2020-12-11    1.155288
## Name: Std_dev, dtype: float64
```


*** `Relative Strength Index` ***

The relative strength index is a indicator of mementum used in technical analysis that measures the magnitude of current price changes to know overbought or oversold conditions in the price of a stock or other asset. If RSI is above 70 then it is overbought. If RSI is below 30 then it is oversold condition.


```python
df['RSI'] = talib.RSI(df['Close'].values, timeperiod = 9)    
print(df[['RSI']].tail())
```

```
##                   RSI
## Date                 
## 2020-12-07  70.157680
## 2020-12-08  67.720435
## 2020-12-09  55.712970
## 2020-12-10  55.874139
## 2020-12-11  57.037305
```

*** `Wiliams %R` ***

Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low of the past N days.The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest). A value of −100 means the close today was the lowest low of the past N days, and 0 means today's close was the highest high of the past N days.


```python
df['Williams %R'] = talib.WILLR(df['High'].values,df['Low'].values, df['Close'].values, 7)
 
print(df[['Williams %R']].tail(7))
```

```
##             Williams %R
## Date                   
## 2020-12-03    -6.832061
## 2020-12-04   -13.016760
## 2020-12-07    -1.923567
## 2020-12-08    -7.795934
## 2020-12-09   -36.764473
## 2020-12-10   -52.071949
## 2020-12-11   -53.193664
```

Readings below -80 represent oversold territory and readings above -20 represent overbought.

*** `ADX` ***

ADX is used to quantify trend strength. ADX calculations are based on a moving average of price range expansion over a given period of time. The average directional index (ADX) is used to determine when the price is trending strongly.

>0-25:	  Absent or Weak Trend

>25-50:	Strong Trend

>50-75:	Very Strong Trend

>75-100:	Extremely Strong Trend


```python

df['ADX'] = talib.ADX(df['High'].values,df['Low'].values, df['Close'].values, 7)


print(df[['ADX']].tail(7))
```

```
##                   ADX
## Date                 
## 2020-12-03  30.401402
## 2020-12-04  27.995900
## 2020-12-07  30.272355
## 2020-12-08  32.438747
## 2020-12-09  29.960885
## 2020-12-10  27.180659
## 2020-12-11  24.120367
```

*** `MACD` ***

Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.


```python
ShortEMA = df.Close.ewm(span=12, adjust=False).mean() #AKA Fast moving average
#Calculate the Long Term Exponential Moving Average
LongEMA = df.Close.ewm(span=26, adjust=False).mean() #AKA Slow moving average
#Calculate the Moving Average Convergence/Divergence (MACD)
MACD = ShortEMA - LongEMA
#Calcualte the signal line
signal = MACD.ewm(span=9, adjust=False).mean()

df['MACD'] = MACD

df['Signal Line'] = signal

df['MACD_IND'] = df['MACD'] - df['Signal Line']

print(df[['MACD_IND']].tail())
```

```
##             MACD_IND
## Date                
## 2020-12-07  0.394404
## 2020-12-08  0.500001
## 2020-12-09  0.433237
## 2020-12-10  0.370972
## 2020-12-11  0.325052
```

*** `Bollinger Bands` ***

Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity.


```python

from talib import MA_Type

upper, middle, lower = talib.BBANDS(df['Close'].values, matype=MA_Type.T3)

print(round(upper[-1],2),round(middle[-1],2),round(lower[-1],2))
```

```
## 65.92 64.18 62.44
```


In case we choose another value than zero for our min_periods parameter, we will end up with a couple of NaN-Values . These are not a number values and they are useless. Therefore, we would want to delete the entries that have such values.


```python
df.dropna( inplace = True )
```

We do this by using the dropna function. If we would have had any entries with NaN values in any column, they would now have been deleted



***3: Predicting the movement of stock***

To predict the movement of the stock we use 5 lag returns as the dependent variables. The first leg is return yesterday, leg2 is return day before yesterday and so on. The dependent variable is whether the prices went up or down on that day. Other variables include the technical indicators which along with 5 lag returns are used to predict the movement of stock using logistic regression.


***3.1: Creating lag returns***



```python

tslagret = pd.DataFrame(index=df.index)
tslagret["Today"] = df["Close"]
no_lags = 6
for i in range(0, no_lags):
    tslagret["Lag%s" % str(i + 1)] = df["Close"].shift(i + 1)
```

***3.2:  Creating returns dataframe***


```python

df_ret = pd.DataFrame(index=tslagret.index)
df_ret["Today"] = tslagret["Today"].pct_change()*100.0
```


***3.2:  create the lagged percentage returns columns***


```python
for i in range(0, no_lags):
    df_ret["Lag%s" % str(i + 1)] = tslagret["Lag%s" % str(i + 1)].pct_change() * 100.0

df_ret.drop(df_ret.index[:7], inplace=True)
```


***3.3: "Direction" column (+1 or -1) indicating an up/down day***


```python
df_ret["Direction"] = np.sign(df_ret["Today"])

df_ret["Direction"] = np.where(df_ret["Direction"] <= 0, -1 , 1)

#df_ret["Direction"] = np.where(df['Close'].shift(-1) > df['Close'], 1,-1)
```

***3.4: Create the dependent and independent variables ***


```python

data = df_ret.copy()

X = data[["Lag1", "Lag2", "Lag3", "Lag4","Lag5","Lag6"]]

X['usdinr'] = df['usdinr'] - 0.5

```

```
## C:/Users/slaxm/AppData/Local/r-miniconda/envs/r-reticulate/python.exe:1: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```

```python
X['usdsilver'] = df['usdsilver']

X["20d_ma"] = df[ '20d_ma' ]

X["50d_ma"] =  df['50d_ma']

X["100d_ma"] =  df['100d_ma']

X["200d_ma"] =  df['200d_ma']

X['Std_dev'] = df['Std_dev']

X['RSI'] = df['RSI']

X['Williams %R'] = df['Williams %R']

X['MACD_IND'] = df['MACD_IND'] 


y = data["Direction"]

print(X.tail())
```

```
##                 Lag1      Lag2      Lag3      Lag4      Lag5  ...  200d_ma  \
## Date                                                          ...            
## 2020-12-07 -0.336620  0.851125  0.379534  4.729204 -2.007193  ...    54.80   
## 2020-12-08  4.847049 -0.336620  0.851125  0.379534  4.729204  ...    54.89   
## 2020-12-09 -0.468709  4.847049 -0.336620  0.851125  0.379534  ...    54.98   
## 2020-12-10 -2.596944 -0.468709  4.847049 -0.336620  0.851125  ...    55.08   
## 2020-12-11  0.048820 -2.596944 -0.468709  4.847049 -0.336620  ...    55.17   
## 
##              Std_dev        RSI  Williams %R  MACD_IND  
## Date                                                    
## 2020-12-07  2.000723  70.157680    -1.923567  0.394404  
## 2020-12-08  2.152803  67.720435    -7.795934  0.500001  
## 2020-12-09  1.456908  55.712970   -36.764473  0.433237  
## 2020-12-10  1.314193  55.874139   -52.071949  0.370972  
## 2020-12-11  1.155288  57.037305   -53.193664  0.325052  
## 
## [5 rows x 16 columns]
```

```python
data.describe()
```

```
##             Today        Lag1        Lag2        Lag3        Lag4        Lag5  \
## count  244.000000  244.000000  244.000000  244.000000  244.000000  244.000000   
## mean     0.160690    0.158399    0.159451    0.172888    0.182393    0.169437   
## std      2.618664    2.618766    2.618773    2.612977    2.614851    2.599430   
## min    -11.221052  -11.221052  -11.221052  -11.221052  -11.221052  -11.221052   
## 25%     -0.878006   -0.878006   -0.878006   -0.855045   -0.855045   -0.855045   
## 50%      0.182187    0.167419    0.182187    0.203833    0.215973    0.215973   
## 75%      1.198904    1.198904    1.198904    1.198904    1.229736    1.229736   
## max      7.031671    7.031671    7.031671    7.031671    7.031671    7.031671   
## 
##              Lag6   Direction  
## count  244.000000  244.000000  
## mean     0.171805    0.106557  
## std      2.599230    0.996350  
## min    -11.221052   -1.000000  
## 25%     -0.855045   -1.000000  
## 50%      0.224232    1.000000  
## 75%      1.229736    1.000000  
## max      7.031671    1.000000
```


***3.5:  Create training and test sets***


```python
start_test = datetime(2020, 8, 1)

X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]
```

***3.6:  Create model ***

```python

# fit model no training data
# model = XGBClassifier()
model = LogisticRegression()
```

***3.7:  train the model on the training set***


```python
model.fit(X_train, y_train)
```

```
## LogisticRegression()
## 
## C:\Users\slaxm\AppData\Local\R-MINI~1\envs\R-RETI~1\lib\site-packages\sklearn\linear_model\_logistic.py:764: ConvergenceWarning: lbfgs failed to converge (status=1):
## STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
## 
## Increase the number of iterations (max_iter) or scale the data as shown in:
##     https://scikit-learn.org/stable/modules/preprocessing.html
## Please also refer to the documentation for alternative solver options:
##     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
##   extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
```

***3.8:    make an array of predictions on the test set***


```python
y_pred = model.predict(X_test)
```

***3.9:    output the hit-rate and the confusion matrix for the model***


```python
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
```

```
## Accuracy:  0.776595744680851
```


```python
print(confusion_matrix(y_pred,y_test))
```

```
## [[30  7]
##  [14 43]]
```

***3.10:   Predict movement of stock for tomorrow. ***


```python
comp = pd.DataFrame()
comp['y_test'] = y_test
comp['y_pred'] = y_pred

print(comp)
```

```
##             y_test  y_pred
## Date                      
## 2020-08-03       1       1
## 2020-08-04       1       1
## 2020-08-05       1       1
## 2020-08-06       1       1
## 2020-08-07      -1      -1
## ...            ...     ...
## 2020-12-07       1       1
## 2020-12-08      -1      -1
## 2020-12-09      -1      -1
## 2020-12-10       1      -1
## 2020-12-11       1      -1
## 
## [94 rows x 2 columns]
```

```python
predict = model.predict(X_test.tail(1))
print(int(predict)) # 1: UP, -1: DOWN
```

```
## -1
```







