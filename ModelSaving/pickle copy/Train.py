import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib

import pickle


from  sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
import statsmodels.regression.linear_model as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score, roc_curve,roc_auc_score,plot_confusion_matrix,classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler

url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'

df=pd.read_csv(url)
print(df)

X=df.iloc[: ,:-1].values
y=df.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=101)

log_reg=LogisticRegression(random_state=42)
log_reg.fit(x_train ,y_train)
y_train_pred= log_reg.predict(x_train)
y_test_pred=log_reg.predict(x_test)

print("train accuracy_score :" , accuracy_score(y_train ,y_train_pred))
print("********" *25)
print("test accuracy: " , accuracy_score(y_test,y_test_pred))

joblib.dump(log_reg,'log_reg.pkl')
pickle.dum(log_reg, open('log_reg.pkl', 'wb'))