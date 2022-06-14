'''
This programme is used for training the Scikit-learn model and then saving it to "model.sav
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import csv

#%matplotlib inline

df = pd.read_csv('AAPL.csv')

df['next_close'] = df['Close'].shift(-1)
df = df.drop(df.index[-1])

X=df.drop('next_close', axis=1)
y=df['next_close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rfr1D = RandomForestRegressor(n_estimators=100)
rfr1D.fit(X_train,y_train)
rfc_pred = rfr1D.predict(X_test)

vali = pd.read_csv('AAPL_pred.csv')
rfr_pred = rfr1D.predict(vali)
print(rfr_pred)

#joblib.dump(rfr1D, 'model1D.sav')
#7-day prediction model
s = []
for x in range(df["Open"].size):
    if x%5 != 0:
        s.append(x)
        
df_7D = df.drop(index=s)

X=df_7D.drop('next_close', axis=1)
y=df_7D['next_close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rfr7D = RandomForestRegressor(n_estimators=100)
rfr7D.fit(X_train,y_train)
rfc_pred = rfr7D.predict(X_test)

vali = pd.read_csv('AAPL_pred.csv')
rfr_pred = rfr7D.predict(vali)
print(rfr_pred)

#28-day prediction model

s = []
for x in range(df["Open"].size):
    if x%20 != 0:
        s.append(x)
        
df_28D = df.drop(index=s)

X=df_28D.drop('next_close', axis=1)
y=df_28D['next_close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rfr28D = RandomForestRegressor(n_estimators=100)
rfr28D.fit(X_train,y_train)
rfc_pred = rfr28D.predict(X_test)

vali = pd.read_csv('AAPL_pred.csv')
rfr_pred = rfr28D.predict(vali)
print(rfr_pred)