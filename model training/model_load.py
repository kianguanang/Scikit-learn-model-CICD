import pandas as pd
import joblib
import json

df_pred = pd.read_csv('AAPL_pred.csv')
df_pred['Difference'] = df_pred['Close'] - df_pred['Open']

loaded_model = joblib.load('model.sav')
result = loaded_model.predict(df_pred)

print(type(result[0]))
output = result[0]
print(result)

price_diff = {
  'n2s':-1.5,
  'n1s':-0.75,
  'sideway':0,
  'p1s':0.75,
  'p2s':1.5
}

prediction_1D = df_pred.at[0,'Close'] + price_diff[output]

print (prediction_1D)

import time

epoch_time = time.time() + 7 * 86400
local_time = time.ctime(epoch_time)

print("The local time is:", local_time)

ret = json.dumps([{local_time: prediction_1D}])
print(ret)
