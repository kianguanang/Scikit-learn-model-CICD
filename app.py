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

    #load the various models
    model_1D = joblib.load('model1D.sav')
    model_7D = joblib.load('model7D.sav')
    model_28D = joblib.load('model28D.sav')

    #prediction
    result1D = model_1D.predict(df)
    result7D = model_7D.predict(df)
    result28D = model_28D.predict(df)

    #extract prediction based on latest date
    output1D = result1D[0]
    output7D = result7D[0]
    output28D = result28D[0]

    #calculate datetime based on request datetime
    latestDate = Date[0]
    dt = datetime.strptime(latestDate, '%Y-%m-%dT%H:%M:%S.%fZ')
    epochdt = dt.timestamp()
    x1D = epochdt + 86400
    x7D = epochdt + 7*86400
    x28D = epochdt + 28*86400

    #JSON string to return
    ret = json.dumps({
      "message":"SUCCESS",
      "result":{
        "ticker_symbol":ticker,
        "model_type":2,
        "prediction":[
          {x1D: output1D},
          {x7D: output7D},
          {x28D: output28D}
        ]}}, indent=4)

    return ret

