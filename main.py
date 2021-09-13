# import libraries

import os
import warnings
import sys

import pandas as pd
import numpy as np
import logging
import pickle
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


# Configure Logger
logging.basicConfig(filename='logfile.log', level=logging.WARN, format = '%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

# load dataset
    try:
        df = pd.read_csv('Admission_Prediction.csv')
    except Exception as e:
        logger.exception('Unable to load dataset, check file path. Error: %s', e)
    else:
        print('Data is loaded correctly')


# Pandas profiling
    pf = ProfileReport(df)
    pf.to_widgets()


# Replacing missing values in 'GRE Score', 'TOEFL Score' and 'University Rating' columns by their means
    df['GRE Score'] = df['GRE Score'].fillna(df['GRE Score'].mean())
    df['TOEFL Score'] = df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
    df['University Rating'] = df['University Rating'].fillna(df['University Rating'].mean())


# check if null values are replaced
    df.isnull().sum()


# descriptive statistics
    df.describe()


# dropping column 'Serial No.'as its not useful
    df.drop(columns='Serial No.', inplace=True)


# storing independent variables as X and dependent variable as y
    X = df.drop(['Chance of Admit'], axis=1)
    y = df['Chance of Admit']


# Normalizing data using StandardScaler
    scaler = StandardScaler()
    arr = scaler.fit_transform(X)


# Check for multicollinearity using Variance Inflation Factor
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_df = pd.DataFrame()
    vif_df['vif'] = [variance_inflation_factor(arr,i) for i in range(arr.shape[1])]
    vif_df['feature'] = X.columns
    vif_df

#Splitting dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(arr,y,test_size=0.25,random_state=100)


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    mlflow.sklearn.autolog()
    with mlflow.start_run():
        elastic = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,max_iter=max_iter,random_state=50)
        elastic.fit(X_train,y_train)
        yhat = elastic.predict(X_test)

        elastic_score = elastic.score(X_test,y_test)
        print("alpha value is", alpha)
        print("L1 ratio is",l1_ratio)
        print("No of iterations",max_iter)
        print("Score is",elastic.score(X_test,y_test))








