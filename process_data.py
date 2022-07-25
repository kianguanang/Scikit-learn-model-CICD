import pandas as pd
import numpy as np
from scipy.stats import norm

with open('ticker.txt', 'r') as f:
  ticker = f.read()

ticker_list = ticker.split(",")

for ticker_sym in ticker_list:
  TICKER = ticker_sym
  DATAFILE_NAME = TICKER+".csv"

  #read data file
  df = pd.read_csv(DATAFILE_NAME)

  df.drop("stockID", axis=1, inplace=True)
  df.drop("Date", axis=1, inplace=True)
  df.drop("DateString", axis=1, inplace=True)
  df_row = df.iloc[[-1]]
  df = pd.concat([df,df_row])

  #data preparation
  #add label - next day's closing price as label for each row, then delete last row which does not have any next closing price
  df['next_close'] = df['Close'].shift(-1)

  #add features
  #add 7-day moving average

  #Calculated 7-day MA using D-6 to D for simplicity (vs D-3 to D+3)
  day = 7
  df['7DMA'] = (df['Close'].rolling(day).sum())/day

  #Calculated 14-period Relative Strength Index (RSI) using D-13 to D
  period = 14
  gain = []
  loss = []

  df_rsi = df.loc[:,'Open':'Close']
  df_rsi = df_rsi.drop('High', axis=1)
  df_rsi = df_rsi.drop('Low', axis=1)
  df_rsi['diff'] = df_rsi['Close'] - df_rsi['Open']
  diff = df_rsi['diff'].tolist()
  for item in diff:
      if item > 0:
          gain.append(item)
          loss.append(0.0)
      else:
          loss.append(item)
          gain.append(0.0)

  df_rsi['gain'] = gain
  df_rsi['loss'] = loss

  total_gain = [0] * len(gain)
  total_loss = [0] * len(gain)
  gain_p = [0] * len(gain)
  loss_p = [0] * len(gain)

  for index, item in enumerate(gain):
      if index<13:
          total_gain[index]=float("nan")
          total_loss[index]=float("nan")
      else:
          for x in range(period):
              total_gain[index]+=gain[index-x]
              total_loss[index]+=loss[index-x]

  df_rsi['total_gain'] = total_gain
  df_rsi['total_loss'] = total_loss

  df_rsi['gain_p'] = df_rsi['total_gain']/df_rsi['Close']
  df_rsi['loss_p'] = df_rsi['total_loss']/df_rsi['Close'] * -1

  df_rsi['rsi'] = 100 - (100/(1+(df_rsi['gain_p']/df_rsi['loss_p'])))
  df['RSI'] = df_rsi['rsi']

  #add 7-Day Rate of Change (ROC_7D)
  df_roc = df
  df_roc['ROC_7D'] = (df_roc['Close']/df_roc['Close'].shift(7) - 1) * 100
  df['ROC_7D'] = df_roc['ROC_7D']

  #add 10-Day annualised Volatility (VOL_10D)
  period = 10
  volatility=[]
  df_vol = df.loc[:,'Open':'next_close']
  df_vol['delta'] = df_vol['next_close']/df_vol['Close'] - 1
  delta = df_vol['delta'].tolist()
  for index, value in enumerate(delta):
      if index<period:
          volatility.append(float("nan"))
      else:
          temp = []
          for x in range(period):
              temp.append(delta[index-x])
          volatility.append(np.std(temp)*15.8745) #15.8745 is the sqrt of 252 average trading days per year

  df['VOLA_10D'] = volatility

  #add 7-Day Disparity Index (DI_7D)
  df_di = df
  df_di['DI_7D'] = 100 * (df_di['Close'] - df_di['7DMA'])/df_di['7DMA']
  df['DI_7D'] = df_di['DI_7D']

  #add Stochastic Oscillator (StochOsc) and Williams%R (willR)
  df_StochOsc = df.loc[:,'Open':'Close']
  period=14
  low = []
  high = []
  so = []
  willR = []
  closing = df_StochOsc['Close'].tolist()
  dayHigh = df_StochOsc['High'].tolist()
  dayLow = df_StochOsc['Low'].tolist()
  for i,v in enumerate(closing):
      if i<14:
          low.append(float('nan'))
          high.append(float('nan'))
          so.append(float('nan'))
          willR.append(float('nan'))
      else:
          tempHigh = []
          tempLow = []
          for x in range(period):
              tempHigh.append(dayHigh[i-x])
              tempLow.append(dayLow[i-x])
          low.append(min(tempLow))
          high.append(max(tempHigh))
          try:
            cal = 100 * (closing[i] - low[i])/(high[i]-low[i])
          except:
            so.append(float('nan'))
          else:
            so.append(cal)
          
          try:
            cal = -100 * (high[i] - closing[i])/ (high[i] - low[i])
          except:
            willR.append(float('nan'))
          else:
            willR.append(cal)

  df_StochOsc['StochOsc'] = so
  df_StochOsc['willR'] = willR
  df['StochOsc'] = df_StochOsc['StochOsc']
  df['willR'] = df_StochOsc['willR']

  #add Volume Price Trend (VPT)
  prev_vpt = [float('nan')]
  vpt = [float('nan')] #first prev_close is NaN so first row not used for calculation
  df_VPT = df
  df_VPT['prev_close'] = df_VPT['Close'].shift(1)
  vpt_close = df_VPT['Close'].tolist()
  vpt_prev_close = df_VPT['prev_close'].tolist()
  vpt_volume = df_VPT['Volume'].tolist()

  initial_VPT = 0 + vpt_volume[1] * (vpt_close[1]-vpt_prev_close[1])/vpt_prev_close[1]
  vpt.append(initial_VPT) #2nd row is the first recorded VPT

  for i in range(len(df_VPT)):
      if i<1:
          pass
      else:
          new_vpt = vpt[i] + vpt_volume[i] * (vpt_close[i]-vpt_prev_close[i])/vpt_prev_close[i]
          vpt.append(new_vpt)

  vpt.pop() #remove last VPT as the row does not exist on the dataframe and will cause index error
  df_VPT['VPT'] = vpt
  df = df.drop('prev_close', axis=1)

  #add Commodity Channel Index (CCI) with period=7day
  df_CCI = df
  sma_list = []
  mean_dev_list = []
  CCI_list = []

  df_CCI['typ_price'] = (df_CCI['High'] + df_CCI['Low'] + df_CCI['Close'])/3
  typ_price_list = df_CCI['typ_price'].tolist()

  for i in range(len(df_CCI)):
      price_temp_list = []
      dev_temp_list = []
      mean_temp = 0.0
      if i<6:
          sma_list.append(float('nan'))
          mean_dev_list.append(float('nan'))
      else:
          for x in range(7):
              price_temp_list.append(typ_price_list[i-x])
          mean_temp = sum(price_temp_list)/7
          sma_list.append(mean_temp)
          for x in range(7):
              dev_temp_list.append(abs(price_temp_list[x] - mean_temp))
          mean_dev_list.append(sum(dev_temp_list)/7)

  for i in range(len(df_CCI)):
    try:
      cci = (typ_price_list[i] - sma_list[i])/(0.015 * mean_dev_list[i])
    except:
      CCI_list.append(float('nan'))
    else:
      CCI_list.append(cci)

  df_CCI['CCI'] = CCI_list
  df = df.drop('typ_price', axis=1)

  #remove rows where result is NaN
  df = df.dropna()

  processed_filename = TICKER + "_proc.csv"
  df.to_csv(processed_filename)