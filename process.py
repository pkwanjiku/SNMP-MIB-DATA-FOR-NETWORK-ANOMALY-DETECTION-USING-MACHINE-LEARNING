import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


df = pd.read_csv('normal.csv', delimiter=';', decimal=',')
print(df.head())
# X = np.asarray(df[[1,"2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34"]])