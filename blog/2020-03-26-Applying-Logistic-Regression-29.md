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

<p class="nocopy">



```{.python .nocopy}
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

</p>

***1.2:Fetching the data***

To fetch the stock data we use `investpy` library. This library fetchs data from investing for example:


```python
df = investpy.commodities.get_commodity_historical_data(stock='ABC',from_date='01/01/2015',to_date='14/08/2020', country = "India")
```



```
##                Open     High      Low    Close  Volume Currency
## Date                                                           
## 2020-01-01  46728.0  46799.0  46315.0  46674.0   21899      INR
## 2020-01-02  46708.0  47202.0  46594.0  47048.0   84481      INR
## 2020-01-03  47100.0  47927.0  47100.0  47540.0  103394      INR
## 2020-01-06  47680.0  48687.0  47361.0  47559.0  111379      INR
## 2020-01-07  47507.0  48190.0  47200.0  48134.0  118498      INR
## ...             ...      ...      ...      ...     ...      ...
## 2021-01-04  68410.0  70600.0  68410.0  69905.0  282309      INR
## 2021-01-05  70110.0  70900.0  70000.0  70777.0  213294      INR
## 2021-01-06  70615.0  71412.0  68600.0  69342.0  360834      INR
## 2021-01-07  69252.0  70198.0  69252.0  69898.0  238639      INR
## 2021-01-08  69838.0  69838.0  63621.0  64348.0  437809      INR
## 
## [264 rows x 6 columns]
```



***1.2.1: Determining position based on volume and close***


```python

pd.set_option('display.max_columns', None)

df2 = df[['Open','High','Low','Close','Volume']]





diffdf = df2.diff()[1:]

print(diffdf.tail(10))
```

```
##               Open    High     Low   Close    Volume
## Date                                                
## 2020-12-28  -273.0  1908.0   745.0  1288.0   54526.0
## 2020-12-29  1250.0  -781.0   251.0  -712.0   -9269.0
## 2020-12-30  -550.0   -43.0   339.0   485.0  -74672.0
## 2020-12-31   451.0  -261.0  -190.0  -452.0  -17233.0
## 2021-01-01  -581.0  -383.0   -85.0   -26.0 -115116.0
## 2021-01-04   390.0  2310.0   645.0  1845.0  254625.0
## 2021-01-05  1700.0   300.0  1590.0   872.0  -69015.0
## 2021-01-06   505.0   512.0 -1400.0 -1435.0  147540.0
## 2021-01-07 -1363.0 -1214.0   652.0   556.0 -122195.0
## 2021-01-08   586.0  -360.0 -5631.0 -5550.0  199170.0
```

```python
df2['Pos'] = np.where((df2['Volume'] > df2['Volume'].shift(1)) & ((df2['Close'] >= df2['Close'].shift(1))),"UP","DOWN")

df2['Str'] = np.where((df2['Open'] > df2['Close'].shift(1)),"Buy","Sell")

df2['SL'] = np.where((df2['Str'] == 'Buy'),df2['Close'].shift(1), df2['High'].shift(1))


print(df2.tail(20))

```

```
##                Open     High      Low    Close  Volume   Pos   Str       SL
## Date                                                                       
## 2020-12-11  63611.0  63890.0  62762.0  63751.0  216360  DOWN   Buy  63531.0
## 2020-12-14  63540.0  64040.0  62775.0  63459.0  224280  DOWN  Sell  63890.0
## 2020-12-15  63550.0  64925.0  63475.0  64826.0  202625  DOWN   Buy  63459.0
## 2020-12-16  64950.0  66720.0  64950.0  65887.0  276515    UP   Buy  64826.0
## 2020-12-17  66187.0  68334.0  66187.0  68183.0  270267  DOWN   Buy  65887.0
## 2020-12-18  68195.0  68280.0  67316.0  67860.0  200090  DOWN   Buy  68183.0
## 2020-12-21  68200.0  71490.0  65299.0  68972.0  437811    UP   Buy  67860.0
## 2020-12-22  69158.0  69654.0  66535.0  66832.0  324111  DOWN   Buy  68972.0
## 2020-12-23  66811.0  67723.0  66300.0  67539.0  252366  DOWN  Sell  69654.0
## 2020-12-24  67723.0  67850.0  66705.0  67477.0  189448  DOWN   Buy  67539.0
## 2020-12-28  67450.0  69758.0  67450.0  68765.0  243974    UP  Sell  67850.0
## 2020-12-29  68700.0  68977.0  67701.0  68053.0  234705  DOWN  Sell  69758.0
## 2020-12-30  68150.0  68934.0  68040.0  68538.0  160033  DOWN   Buy  68053.0
## 2020-12-31  68601.0  68673.0  67850.0  68086.0  142800  DOWN   Buy  68538.0
## 2021-01-01  68020.0  68290.0  67765.0  68060.0   27684  DOWN  Sell  68673.0
## 2021-01-04  68410.0  70600.0  68410.0  69905.0  282309    UP   Buy  68060.0
## 2021-01-05  70110.0  70900.0  70000.0  70777.0  213294  DOWN   Buy  69905.0
## 2021-01-06  70615.0  71412.0  68600.0  69342.0  360834  DOWN  Sell  70900.0
## 2021-01-07  69252.0  70198.0  69252.0  69898.0  238639  DOWN  Sell  71412.0
## 2021-01-08  69838.0  69838.0  63621.0  64348.0  437809  DOWN  Sell  70198.0
```



***1.2.2: Determining Average monthly closing prices***


```
##               Open    High     Low   Close  Volume
## Date                                              
## 2020-01-01  46.728  46.799  46.315  46.674   21899
## 2020-01-02  46.708  47.202  46.594  47.048   84481
## 2020-01-03  47.100  47.927  47.100  47.540  103394
## 2020-01-06  47.680  48.687  47.361  47.559  111379
## 2020-01-07  47.507  48.190  47.200  48.134  118498
## ...            ...     ...     ...     ...     ...
## 2021-01-04  68.410  70.600  68.410  69.905  282309
## 2021-01-05  70.110  70.900  70.000  70.777  213294
## 2021-01-06  70.615  71.412  68.600  69.342  360834
## 2021-01-07  69.252  70.198  69.252  69.898  238639
## 2021-01-08  69.838  69.838  63.621  64.348  437809
## 
## [264 rows x 5 columns]
```

```
## Date
## 2020-01-31    46.715435
## 2020-02-29    46.769810
## 2020-03-31    42.008682
## 2020-04-30    43.175278
## 2020-05-31    46.064714
## 2020-06-30    48.667409
## 2020-07-31    55.923957
## 2020-08-31    68.698238
## 2020-09-30    65.459455
## 2020-10-31    61.675571
## 2020-11-30    62.081364
## 2020-12-31    65.897045
## 2021-01-31    68.721667
## Freq: M, Name: Close, dtype: float64
```

```
## meanprice is  54.87906439393939
```


***1.2.3: Technical moving averages***


```python

data = investpy.moving_averages(name='MCX Silver Micro', country='India', product_type='commodity', interval='daily')
print(data.head())
```

```
##   period  sma_value sma_signal   ema_value ema_signal
## 0      5   68770.00       sell  67641.9077       sell
## 1     10   68535.20       sell  68004.1237       sell
## 2     20   67506.90       sell  67225.6626       sell
## 3     50   64548.70       sell  65637.9593       sell
## 4    100   64262.53       sell  62938.1307        buy
```

***1.2.4: Determining the z-scores***


```python

df['zscore'] = stats.zscore(df['Close'])

print(df['zscore'].tail(30))
```

```
## Date
## 2020-11-27    0.556071
## 2020-11-30    0.536449
## 2020-12-01    0.831393
## 2020-12-02    0.846789
## 2020-12-03    0.877783
## 2020-12-04    0.895393
## 2020-12-07    1.066363
## 2020-12-08    1.032149
## 2020-12-09    0.872551
## 2020-12-10    0.870639
## 2020-12-11    0.892777
## 2020-12-14    0.863393
## 2020-12-15    1.000954
## 2020-12-16    1.107721
## 2020-12-17    1.338766
## 2020-12-18    1.306263
## 2020-12-21    1.418163
## 2020-12-22    1.202816
## 2020-12-23    1.273961
## 2020-12-24    1.267722
## 2020-12-28    1.397333
## 2020-12-29    1.325684
## 2020-12-30    1.374490
## 2020-12-31    1.329005
## 2021-01-01    1.326389
## 2021-01-04    1.512050
## 2021-01-05    1.599799
## 2021-01-06    1.455396
## 2021-01-07    1.511346
## 2021-01-08    0.952853
## Name: zscore, dtype: float64
```


***1.2.5: Determining the daily support and resistance levels***


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
## 2021-01-04  69.638333  70.866667  68.676667  71.828333  67.448333  73.056667   
## 2021-01-05  70.559000  71.118000  70.218000  71.459000  69.659000  72.018000   
## 2021-01-06  69.784667  70.969333  68.157333  72.596667  66.972667  73.781333   
## 2021-01-07  69.782667  70.313333  69.367333  70.728667  68.836667  71.259333   
## 2021-01-08  65.935667  68.250333  62.033333  72.152667  59.718667  74.467333   
## 
##                    S3  
## Date                   
## 2021-01-04  66.486667  
## 2021-01-05  69.318000  
## 2021-01-06  65.345333  
## 2021-01-07  68.421333  
## 2021-01-08  55.816333
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
## 52.701
```


But we could also use simple indexing to access certain positions.


```python
print (df[ 'Close' ][ 5 ])
```

```
## 47.426
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


```python
df[["PCT_Change"]].describe()
```

```
##        PCT_Change
## count  264.000000
## mean     0.001234
## std      0.024244
## min     -0.107200
## 25%     -0.007793
## 50%      0.002471
## 75%      0.011362
## max      0.062274
```



```python
print(df[["Close"]].tail(1))
```

```
##              Close
## Date              
## 2021-01-08  64.348
```

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
## 2021-01-04  68.410  70.600  68.410  69.905  282309  ...  73.070    69905.0   
## 2021-01-05  70.110  70.900  70.000  70.777  213294  ...  73.120    70777.0   
## 2021-01-06  70.615  71.412  68.600  69.342  360834  ...  73.118    69342.0   
## 2021-01-07  69.252  70.198  69.252  69.898  238639  ...  73.420    69898.0   
## 2021-01-08  69.838  69.838  63.621  64.348  437809  ...  73.330    64348.0   
## 
##               zscore  PCT_Change    HL_PCT  
## Date                                        
## 2021-01-04  1.512050    0.021854  0.031328  
## 2021-01-05  1.599799    0.009514  0.012716  
## 2021-01-06  1.455396   -0.018027  0.040553  
## 2021-01-07  1.511346    0.009328  0.013534  
## 2021-01-08  0.952853   -0.078610  0.096615  
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
## 2020-12-31  68.086  68.18   66.17   63.67    64.27    57.08
## 2021-01-01  68.060  68.30   66.39   63.78    64.24    57.24
## 2021-01-04  69.905  68.53   66.69   63.94    64.27    57.40
## 2021-01-05  70.777  69.07   66.96   64.11    64.28    57.55
## 2021-01-06  69.342  69.23   67.17   64.29    64.28    57.68
## 2021-01-07  69.898  69.60   67.49   64.49    64.30    57.83
## 2021-01-08  64.348  68.85   67.53   64.56    64.27    57.94
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
## 2020-12-31    0.666225
## 2021-01-01    0.471910
## 2021-01-04    0.774900
## 2021-01-05    1.061925
## 2021-01-06    1.073531
## 2021-01-07    1.037955
## 2021-01-08    2.133321
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
## 2021-01-04  69.909560
## 2021-01-05  73.308013
## 2021-01-06  60.630555
## 2021-01-07  63.390202
## 2021-01-08  35.469729
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
## 2020-12-31   -48.351648
## 2021-01-01   -49.103528
## 2021-01-04   -17.843389
## 2021-01-05    -3.565217
## 2021-01-06   -55.780113
## 2021-01-07   -41.513573
## 2021-01-08   -90.668720
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
## 2020-12-31  62.817443
## 2021-01-01  63.420397
## 2021-01-04  65.914626
## 2021-01-05  68.215931
## 2021-01-06  63.800084
## 2021-01-07  60.015072
## 2021-01-08  57.362494
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
## 2021-01-04  0.135087
## 2021-01-05  0.207348
## 2021-01-06  0.137005
## 2021-01-07  0.107717
## 2021-01-08 -0.284219
```

*** `Bollinger Bands` ***

Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity.


```python

from talib import MA_Type

upper, middle, lower = talib.BBANDS(df['Close'].values, matype=MA_Type.T3)

print(round(upper[-1],2),round(middle[-1],2),round(lower[-1],2))
```

```
## 73.79 69.19 64.59
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
## 2021-01-04 -0.038187 -0.659488  0.712680 -1.035410  1.908799  ...    57.40   
## 2021-01-05  2.710843 -0.038187 -0.659488  0.712680 -1.035410  ...    57.55   
## 2021-01-06  1.247407  2.710843 -0.038187 -0.659488  0.712680  ...    57.68   
## 2021-01-07 -2.027495  1.247407  2.710843 -0.038187 -0.659488  ...    57.83   
## 2021-01-08  0.801823 -2.027495  1.247407  2.710843 -0.038187  ...    57.94   
## 
##              Std_dev        RSI  Williams %R  MACD_IND  
## Date                                                    
## 2021-01-04  0.774900  69.909560   -17.843389  0.135087  
## 2021-01-05  1.061925  73.308013    -3.565217  0.207348  
## 2021-01-06  1.073531  60.630555   -55.780113  0.137005  
## 2021-01-07  1.037955  63.390202   -41.513573  0.107717  
## 2021-01-08  2.133321  35.469729   -90.668720 -0.284219  
## 
## [5 rows x 16 columns]
```

```python
data.describe()
```

```
##             Today        Lag1        Lag2        Lag3        Lag4        Lag5  \
## count  242.000000  242.000000  242.000000  242.000000  242.000000  242.000000   
## mean     0.177501    0.196858    0.194161    0.207454    0.203622    0.193190   
## std      2.600534    2.556929    2.556632    2.553394    2.552523    2.547387   
## min    -11.142960  -11.142960  -11.142960  -11.142960  -11.142960  -11.142960   
## 25%     -0.741985   -0.741985   -0.741985   -0.730271   -0.730271   -0.730271   
## 50%      0.251184    0.251184    0.230260    0.251184    0.251184    0.230260   
## 75%      1.422146    1.422146    1.422146    1.422146    1.422146    1.360934   
## max      7.004323    7.004323    7.004323    7.004323    7.004323    7.004323   
## 
##              Lag6   Direction  
## count  242.000000  242.000000  
## mean     0.188457    0.090909  
## std      2.548882    0.997923  
## min    -11.142960   -1.000000  
## 25%     -0.741985   -1.000000  
## 50%      0.230260    1.000000  
## 75%      1.360934    1.000000  
## max      7.004323    1.000000
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
## Accuracy:  0.6637168141592921
```


```python
print(confusion_matrix(y_pred,y_test))
```

```
## [[16  1]
##  [37 59]]
```

***3.10:   Predict movement of stock for tomorrow. ***


```python
comp = pd.DataFrame()
comp['y_test'] = y_test
comp['y_pred'] = y_pred

print(comp.tail(20))
```

```
##             y_test  y_pred
## Date                      
## 2020-12-11       1       1
## 2020-12-14      -1       1
## 2020-12-15       1       1
## 2020-12-16       1       1
## 2020-12-17       1       1
## 2020-12-18      -1       1
## 2020-12-21       1       1
## 2020-12-22      -1       1
## 2020-12-23       1       1
## 2020-12-24      -1       1
## 2020-12-28       1       1
## 2020-12-29      -1       1
## 2020-12-30       1       1
## 2020-12-31      -1       1
## 2021-01-01      -1       1
## 2021-01-04       1       1
## 2021-01-05       1       1
## 2021-01-06      -1       1
## 2021-01-07       1       1
## 2021-01-08      -1      -1
```

```python
predict = model.predict(X_test.tail(1))
print(int(predict)) # 1: UP, -1: DOWN
```

```
## -1
```



