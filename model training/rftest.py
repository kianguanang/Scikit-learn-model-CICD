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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import csv

#%matplotlib inline

df = pd.read_csv('AAPL.csv')

df['Difference'] = df['Close'] - df['Open']

conditions = [
    (df['Difference'] <= -1.5),
    (df['Difference'] <0.125) & (df['Difference'] > -1.5),
    (df['Difference'] <= -0.125) & (df['Difference'] < 0.125),
    (df['Difference'] >= 0.125) & (df['Difference'] < 1.5),
    (df['Difference'] >= 1.5)
    ]

values = ['n2s', 'n1s', 'sideway', 'p1s', 'p2s']
df['deviation'] = np.select(conditions, values)
df.info()
print(df['deviation'])

X=df.drop('deviation', axis=1)
y=df['deviation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rfc = RandomForestClassifier(n_estimators=32)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test, rfc_pred))

joblib.dump(rfc, 'model.sav')