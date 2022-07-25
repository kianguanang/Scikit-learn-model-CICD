import requests as requests
import pandas as pd

stock_list = ["AAPL", "AMCR"]#, "SLV", "VAW", "GLTR", "SPY", "VGT", "VHT", "VNQ", "SNPS", "WMT", "GLD"]

temp_str = ""
for i in range(len(stock_list)):
  if i == len(stock_list)-1:
    temp_str = temp_str + stock_list[i]
  else:
    temp_str = temp_str + stock_list[i] + ","

with open("ticker.txt", "w") as f:
  f.write(temp_str)

with open('ticker.txt', 'r') as f:
  ticker = f.read()
ticker_list = ticker.split(",")

def test(tick):
  __url = 'https://algotuno-web3.vercel.app/api/stock/get_hsp'
  __headers = {'Authorization': 'Bearer 9ddf045fa71e89c6d0d71302c0c5c97e'}
  __body = {
	"ticker_symbol" : 	tick,
	"start_date"	:	"",
	"end_date"	:	"",
	"sort"		:	"asc"
}
  return requests.post(
    url=__url,
    headers=__headers,
    data=__body,
  )

for i in range(len(stock_list)):
  fileName = stock_list[i] + ".csv"

  json_stock = test(stock_list[i]).json()

  df = pd.DataFrame(json_stock['results'])


  df.to_csv(fileName, index=False)