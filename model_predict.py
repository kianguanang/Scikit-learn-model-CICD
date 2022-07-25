import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import joblib
import time
import boto3
from botocore.exceptions import ClientError
from cmath import sqrt
import scipy
import os

st = time.time()

""" Set arrays of all features (n) and number of features to include (r). 
Total number of combinations evaluated will be nCr to find the one 
with the lowest MAPE followed by RMSE """
#featureArray = ["Open", "High", "Low", "Close", "Volume", "7DMA", "RSI", "ROC_7D", 'VOLA_10D', 'DI_7D', 'StochOsc', 'willR', 'VPT', 'CCI']
featureArray = ["Open", "High", "Low", "Close", "Volume", "7DMA", "RSI", "ROC_7D", 'VOLA_10D', 'DI_7D']
MIN_FEATURE_COUNT = 8  #state the min number of features to use in the model
MAX_FEATURE_COUNT = 9  #state the max number of features to use in the model
FEATURE_COUNT_RANGE = range(MIN_FEATURE_COUNT,MAX_FEATURE_COUNT+1)
ITERATION = 1
NO_OF_TREES = 64
TEST_PERCENTAGE = 0.2
MEASUREMENT_METRIC = "MAPE"  #or "RMSE"

def Combination(inputArray, combinationArray, n, r, index, i, arrayList):
  if index == r:
    temp_arr = []
    for item in combinationArray:
      #print(item, end = " ")
      temp_arr.append(item)
    arrayList.append(temp_arr)
    #print()
    return
  if i >= n:
    return
  combinationArray[index] = inputArray[i]
  Combination(inputArray, combinationArray, n, r, index + 1, i + 1, arrayList)
  Combination(inputArray, combinationArray, n, r, index, i + 1, arrayList)

#clear the previous report version
f = open("report.txt",'w')
f.close()

with open('ticker.txt', 'r') as f:
  ticker = f.read()
ticker_list = ticker.split(",")

df_prediction = pd.DataFrame()

for ticker_sym in ticker_list:
  TICKER = ticker_sym
  DATAFILE_NAME = TICKER+"_proc.csv"
  prediction_list = [0]*3

  #read data file
  df = pd.read_csv(DATAFILE_NAME)

  #7-day prediction model by forecasting using weekly data
  s = []
  for x in range(df["Open"].size):
      if x<=13:
        pass
      elif x%5 != 0:
        s.append(x)
          
  df_7D = df.drop(index=s)

  #30-day prediction model by forecasting using monthly data

  s = []
  for x in range(df["Open"].size):
      if x<=13:
        pass
      elif x%20 != 0:
        s.append(x)
          
  df_30D = df.drop(index=s)

  dataset = [df, df_7D, df_30D]
  day_list = ["1D", "7D", "30D"]

  for d in range(3):
    day_str = day_list[d]
    main_df = pd.DataFrame() #dataframe to capture the record with the absolute lowest mape
    abs_low_mape = 1000.0
    counter = 0
    combination_count = 0
    for k in FEATURE_COUNT_RANGE:
      FEATURE_COUNT=k
      p = os.path.dirname(__file__)
      rel_path = "/combination_results/"
      path1 = p+rel_path
      os.makedirs(path1, exist_ok=True)
      fileName = str("{}{}_m{}_file{}.csv".format(path1, TICKER, d, FEATURE_COUNT))
      modelName = str("{}_model_{}.sav".format(TICKER, d)) #0 for 1d, 1 for 7d, 2 for 30d
      n = len(featureArray)
      r = FEATURE_COUNT
      combinationArray = [""] * r
      arrayList = []
      Combination(featureArray, combinationArray, n, r, 0, 0, arrayList)

      feature_list = []
      for i in range(FEATURE_COUNT):
        temp_feat = "Feature {}".format(i)
        temp_corr = "Correlation {}".format(i)
        feature_list.append(temp_feat)
        feature_list.append(temp_corr)
      feature_list.append("MSE")
      feature_list.append("RMSE")
      feature_list.append("MAPE")

      df_test = pd.DataFrame()
      df_feature_6 = pd.DataFrame()
      feature_df = pd.DataFrame()

      for x in range(len(arrayList)):       #n in nCr
        for y in range(len(arrayList[0])):  #r in nCr
          df_test[arrayList[x][y]] = dataset[d][arrayList[x][y]] #select features from main dataframe into test dataframe
        df_test['next_close'] = df['next_close']
        
        #extract latest data and save it for prediction later
        df_pred = df_test.iloc[[-1]]
        df_pred = df_pred.drop('next_close', axis=1)
        df_test.drop(df_test.index[-1], inplace=True)

        #set training and test data
        X=df_test.drop('next_close', axis=1)
        y=df_test['next_close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENTAGE)

        min_mape = 1000.0
        for j in range(ITERATION):
          #create model and set parameters
          rfr1D = RandomForestRegressor(n_estimators=NO_OF_TREES, n_jobs=-1)
          rfr1D.fit(X_train,y_train)
          
          y_pred = rfr1D.predict(X_test)
          y_pred = pd.Series(y_pred)

          mse = mean_squared_error(y_test, y_pred)
          rmse = mean_squared_error(y_test, y_pred, squared=False)
          
          n_mape=len(y_test)
          sum_mape = 0.0
          for p in range(n_mape):
            temp_num = abs(y_test.iloc[p] - y_pred.iloc[p])/y_test.iloc[p]
            sum_mape+=temp_num
          mape = sum_mape/n_mape

          importance = rfr1D.feature_importances_
          col_name = X.columns.values
          col_imp = []
          combination_count+=1

          if mape<min_mape:
            min_mape = mape

            for i,v in enumerate(importance):
              col_imp.append(col_name[i])
              col_imp.append(v)

            col_imp.append(mse)
            col_imp.append(rmse)
            col_imp.append(mape)
            feature_dict = {i:[v] for i,v in enumerate(col_imp)} #convert list to dict
            feature_df = pd.DataFrame(feature_dict) #convert dict to dataframe to concat
            
            if mape<abs_low_mape:
              abs_low_mape = mape
              #joblib.dump(rfr1D, modelName)
              counter += 1
              print("model built {} time".format(counter)) #build model with lowest mape
              
              #predict next day price using latest data
              prediction = rfr1D.predict(df_pred)
              print(prediction)


              temp_dict = {i:[v] for i,v in enumerate(col_imp)} #convert list to dict
              main_df = pd.DataFrame(temp_dict) #convert dict to dataframe
              main_df.columns = feature_list
        
        df_feature_6 = pd.concat([df_feature_6, feature_df])
        df_test = pd.DataFrame()
        prediction_list[d] = prediction[0]

      df_feature_6.columns = feature_list

      df_feature_6.sort_values(by=[MEASUREMENT_METRIC], inplace=True)

      df_feature_6.to_csv(fileName)
      print(df_feature_6.head(6))

    et = time.time()
    pt = et - st

    print(main_df)
    #Print result of model optimisation into a report
    divider1 = "\n================================================"
    divider2 = "\n-----------------------------------------------------------------------------------------------"
    str_1 = str("\nThis prediction model uses the Random Forests Regression Model with {} trees".format(NO_OF_TREES))
    str_2 = str("\nModel was optimised for {} records and {} possible features(n)".format(len(df.index), len(featureArray)))
    str_3 = str("\nBetween {} and {} features were evaluated(r)".format(MIN_FEATURE_COUNT, MAX_FEATURE_COUNT))
    str_4 = str("\nTotal number of feature combinations (nCr) evaluated: {} (sum of {}C{} to {}C{})".format(int(combination_count/ITERATION), len(featureArray), MIN_FEATURE_COUNT, len(featureArray), MAX_FEATURE_COUNT))
    str_5 = str("\nEach combination was iterated {} times and the iteration with the lowest error was selected".format(ITERATION))
    str_6 = "\nThe optimal model consists of the following features and the corresponding correlations:\n"
    str_7 = ""
    str_8 = "\nOptimised error rate: MAPE = {}".format(round(main_df.loc[0,'MAPE'],3))
    str_9 = str("\n\nTotal processing time: {} seconds".format(int(pt)))
    col_num = int((len(main_df.columns)-3)/2)
    for l in range(col_num):
      temp = str("{}:\t{}".format(main_df.iloc[0,l*2], round(main_df.iloc[0,l*2+1],4)))
      str_7 +="\n"+temp

    #calculate and print confidence
    list1 = y_test
    cf = main_df.loc[0,'MAPE'] * prediction[0]
    n = len(list1)
    s = statistics.mean(list1)
    stdv = statistics.stdev(list1)
    z = abs(cf*sqrt(n)/s)
    p = abs(round(scipy.stats.norm.cdf(z),4))
    str_10 = str("\ncf={}, n={}, s={}, std={}, z={}, confidence={}".format(cf, n, s, stdv, z, p))

    report = TICKER + " " + day_str + divider2 + str_1 + str_2 + str_3 + str_4 + str_5 + divider2 + str_6 + str_7 + divider1 + str_8 + divider1 + str_10 + str_9
    print(report)

    f = open("report.txt", "a")
    f.write(report)
    f.write("\n\n")
    f.close()

    prediction_list.append(p)
    prediction_list.append(main_df.loc[0,'MAPE'])


  try:
    df_prediction[TICKER] = prediction_list
  except:
    pass
  else:
    print(df_prediction.head(10))

df_prediction.to_csv("prediction.csv")

#Send email to notify upon complete model optimisation
SENDER = "Kian Guan <kianguan@hotmail.com>"
RECIPIENT = "kianguanang@gmail.com"
#CONFIGURATION_SET = "ConfigSet"

AWS_REGION = "us-east-1"
SUBJECT = "Prediction Completed"
BODY_TEXT = ("predictions completed")
CHARSET = "UTF-8"

client = boto3.client('ses', region_name=AWS_REGION)

try:
  response = client.send_email(
    Destination={
      'ToAddresses':[
        RECIPIENT,
      ],
    },
    Message={
      'Body': {
        'Text':{
          'Charset': CHARSET,
          'Data': BODY_TEXT,
        },
      },
      'Subject':{
        'Charset':CHARSET,
        'Data': SUBJECT,
      },
    },
    Source=SENDER,
    #ConfigurationSetName=CONFIGURATION_SET,
  )
except ClientError as e:
  print(e.response['Error']['Message'])
else:
  print("Email sent! Message ID:")
  print(response['MessageId'])
