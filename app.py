import requests as requests
import json
import pandas as pd
import joblib
import datetime

def lambda_handler(event, context):
    
    #data sent to aws lambda is captured under "body" of the event object, need to extract it out for subsequent processing
    body = event["body"]
    
    body_dict = json.loads(body)
    
    ticker = body_dict["ticker_symbol"]
    data = body_dict["stock_metadata_list"]

    Open = []
    High = []
    Low = []
    Close = []
    Volume =[]

    for item in data:
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
    x1D = datetime.datetime.now() + datetime.timedelta(days=1)
    x7D = datetime.datetime.now() + datetime.timedelta(days=7)
    x28D = datetime.datetime.now() + datetime.timedelta(days=28)

    #JSON string to return
    ret = json.dumps({
      "message":"SUCCESS",
      "result":{
        "ticker_symbol":ticker,
        "model_type":2,
        "prediction":[
          {x1D.ctime(): output1D},
          {x7D.ctime(): output7D},
          {x28D.ctime(): output28D}
        ]}}, indent=4)

    return ret

