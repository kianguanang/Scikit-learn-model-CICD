import requests as requests
import pandas as pd
import json
import joblib
import time

def test():
  __url = 'https://algotuno-web3.vercel.app/api/stock/get_hsp'
  __headers = {'Authorization': 'Bearer 9ddf045fa71e89c6d0d71302c0c5c97e'}
  __body = {
	"ticker_symbol" : 	"TSLA",
	"start_date"	:	"",
	"end_date"	:	"",
	"sort"		:	"desc"
}
  return requests.post(
    url=__url,
    headers=__headers,
    data=__body,
  )

'''
json_stock = {
  "message": "Found 1259 records for TSLA", 
  "results": [
    {"stockID": 1, "Date": "2022-06-09T00:00:00.000Z", "DateString": "9-JUN-2022", "Open": 748.02002, "High": 748.69989, "Low": 740.51001, "Close": 741, "Volume": 2573250}, 
    {"stockID": 1, "Date": "2022-06-08T00:00:00.000Z", "DateString": "8-JUN-2022", "Open": 720.26001, "High": 749.890015, "Low": 717.530029, "Close": 725.599976, "Volume": 25339600}, 
    {"stockID": 1, "Date": "2022-06-06T00:00:00.000Z", "DateString": "6-JUN-2022", "Open": 733.059998, "High": 734.599976, "Low": 703.049988, "Close": 714.840027, "Volume": 28068200}, 
    {"stockID": 1, "Date": "2022-06-03T00:00:00.000Z", "DateString": "3-JUN-2022", "Open": 729.674988, "High": 743.389893, "Low": 700.253418, "Close": 703.549988, "Volume": 37464579}]}

test().json() returns a dictionary object
{
  "message": value
  "results":[]
}
'''

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

#print(type(ret))
#print(ret)


# https://www.w3schools.com/python/ref_requests_post.asp