import requests as requests
import pandas as pd
import json
from datetime import datetime

df = pd.read_csv("prediction.csv")

f = open("ticker.txt", "r")
ticker = f.read()
ticker_list = ticker.split(",")

#JSON string to return
for ticker in ticker_list:
  
  #format datetime for output
  fileName = ticker+".csv"
  df_date = pd.read_csv(fileName)
  date_list = df_date["Date"]
  latestDate = date_list.iloc[-1]

  #get epoch time in milliseconds
  dt = datetime.strptime(latestDate, '%Y-%m-%dT%H:%M:%S.%fZ')
  epochdt = dt.timestamp()
  x1D = int(epochdt + 86400 + 28800) * 1000
  x7D = int(epochdt + 7*86400 + 28800) * 1000
  x30D = int(epochdt + 30*86400 + 28800) * 1000

  
  price_1D = round(df[ticker][0],2)
  price_7D = round(df[ticker][1],2)
  price_30D = round(df[ticker][2],2)

  cf_1D = round(df[ticker][3]*100,2)
  cf_7D = round(df[ticker][5]*100,2)
  cf_30D = round(df[ticker][7]*100,2)

  err_1D = round(df[ticker][4]*100,2)
  err_7D = round(df[ticker][6]*100,2)
  err_30D = round(df[ticker][8]*100,2)

  res = requests.post(
    url='https://algotuno-web3.vercel.app/api/stock/update_ml_prices',
    json={
      "ticker_symbol":ticker,
      "model_type":"2",
      "prediction":[
        {
          "epoch_time": x1D,
          "price": price_1D,
          "confidence_score": cf_1D,
          "rate_of_error": err_1D,
        },
        {
          "epoch_time": x7D,
          "price": price_7D,
          "confidence_score": cf_7D,
          "rate_of_error": err_7D,
        },
        {
          "epoch_time": x30D,
          "price": price_30D,
          "confidence_score": cf_30D,
          "rate_of_error": err_30D,
        },
      ],
    },
  )

  #res = update_ml_prices(json_obj)
  #print(json_obj)
  #print("res: ", res.json())