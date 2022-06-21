import requests as requests
import json
import pandas as pd
import joblib
import time

def lambda_handler(event, context):
    def test():
        __url = 'https://algotuno-web3.vercel.app/api/stock/get_hsp'
        __headers = {'Authorization': 'Bearer 9ddf045fa71e89c6d0d71302c0c5c97e'}
        __body = {
        "ticker_symbol" :     "TSLA",
        "start_date"    :    "",
        "end_date"    :    "",
        "sort"        :    "desc"
        }
        return requests.post(
        url=__url,
        headers=__headers,
        data=__body,
        )

    json_stock = test().json()

    pre_data = json.dumps(json_stock["results"])
    data = json.loads(pre_data)

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
    #df.info()
    #print(df)
    df['Difference'] = df['Close'] - df['Open']

    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(df)

    output1D = result[0]
    output7D = result[6]
    output28D = result[27]

    price_diff = {
      'n2s':-1.5,
      'n1s':-0.75,
      'sideway':0,
      'p1s':0.75,
      'p2s':1.5
    }

    prediction_1D = df.at[0,'Close'] + price_diff[output1D]
    prediction_7D = df.at[4,'Close'] + price_diff[output7D]
    prediction_28D = df.at[27,'Close'] + price_diff[output28D]

    #print (prediction_1D)

    epoch_time_1D = time.time() + 86400
    epoch_time_7D = time.time() + 7 * 86400
    epoch_time_28D = time.time() + 28 * 86400

    local_time_1D = time.ctime(epoch_time_1D)
    local_time_7D = time.ctime(epoch_time_7D)
    local_time_28D = time.ctime(epoch_time_28D)

    ret = json.dumps([
      {local_time_1D: prediction_1D},
      {local_time_7D: prediction_7D},
      {local_time_28D: prediction_28D}
      ], indent=4)
      
      
    return ret

    '''
    output must be able to convert into json format
    '''
