import requests as requests
import json
import pandas as pd
import joblib
from datetime import datetime

def lambda_handler(event, context):
    
    #data sent to aws lambda is captured under "body" of the event object, need to extract it out for subsequent processing
    body = event["body"]
    
    body_dict = json.loads(body)
    
    ticker = body_dict["ticker_symbol"]
    data = body_dict["stock_metadata_list"]

    #list of trained models
    model1D_list = ['AAPL_1D.sav', 'AMZN_1D.sav', 'AMCR_1D.sav', 'TSLA_1D.sav']
    model7D_list = ['AAPL_7D.sav', 'AMZN_7D.sav', 'AMCR_7D.sav', 'TSLA_7D.sav']
    model28D_list = ['AAPL_28D.sav', 'AMZN_28D.sav', 'AMCR_28D.sav', 'TSLA_28D.sav']

    #directory in current folder storing trained models
    dir = 'model/'

    #index of stocks
    symbol = {
      "AAPL": 0,
      "AMZN": 1,
      "AMCR": 2,
      "TSLA": 3
    }

    ticker_index = symbol[ticker]

    #load the various models
    model_1D = joblib.load(dir + model1D_list[ticker_index])
    model_7D = joblib.load(dir + model7D_list[ticker_index])
    model_28D = joblib.load(dir + model28D_list[ticker_index])

    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Volume =[]

    for item in data:
      Date.append(item['Date'])
      Open.append(item['Open'])
      High.append(item['High'])
      Low.append(item['Low'])
      Close.append(item['Close'])
      Volume.append(item['Volume'])

    dataTable = {
      'Open': Open,
      'High': High,
      'Low': Low,
      'Close': Close,
      'Volume': Volume
    }

    df = pd.DataFrame(dataTable)

    #prediction
    result1D = model_1D.predict(df)
    result7D = model_7D.predict(df)
    result28D = model_28D.predict(df)

    #extract prediction based on latest date
    output1D = result1D[-1]
    output7D = result7D[-1]
    output28D = result28D[-1]

    #format datetime for output
    latestDate = Date[-1]
    dt = datetime.strptime(latestDate, '%Y-%m-%dT%H:%M:%S.%fZ')
    epochdt = dt.timestamp()
    x1D = int(epochdt + 86400 + 28800) * 1000
    x7D = int(epochdt + 7*86400 + 28800) * 1000
    x28D = int(epochdt + 28*86400 + 28800) * 1000

    #JSON string to return
    ret = json.dumps({
      "message":"This is a test message v1",
      "result":{
        "ticker_symbol":ticker,
        "model_type":2,
        "prediction":[
          {x1D: output1D},
          {x7D: output7D},
          {x28D: output28D}
        ]}}, indent=4)

    return ret

