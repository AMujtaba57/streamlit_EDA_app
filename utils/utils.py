import warnings
warnings.filterwarnings('ignore')
import numpy
import pandas as pd
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


def read_data(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    return read_csv(filename, delim_whitespace=True, names=names)

def missing_value(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    missing_values = dataframe.isnull()
    return missing_values.sum()

def show_datatype(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    return dataframe.dtypes

def descriptive_search(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    return dataframe.describe()

def normailze_dataframe(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    return pd.DataFrame(normalize(dataframe), columns=names)

def standardize_dataframe(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    # standardize the dataframe
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(dataframe), columns=names)


def minmax_dataframe(filename):
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    minmax_scaler = MinMaxScaler()
    return pd.DataFrame(minmax_scaler.fit_transform(dataframe), columns=names)

def perform_prediction(filename, model_name, transform):
    if transform == "StandardScaler":
        dataframe = standardize_dataframe(filename)
    elif transform == "Normalize":
        dataframe = normailze_dataframe(filename)
    elif transform == "MinMaxScaler":
        dataframe = minmax_dataframe(filename)
    else:
        dataframe = read_data(filename)

    array = dataframe.values
    X = array[:,0:13]
    Y = array[:,13]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    num_folds = 10

    scoring = 'neg_mean_squared_error'
    
    if model_name == "LogisticRegression":
        name, model = 'LR', LinearRegression()
    elif model_name == "SVM":
        name, model =  'SVR', SVR()
    elif model_name == "ElasticNet":
        name, model = 'EN', ElasticNet()
    elif model_name == "Lasso":
        name, model = 'LASSO', Lasso()
    else:
        name, model = 'CART', DecisionTreeRegressor()

    kfold = KFold(n_splits=num_folds)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    msg = {
        "Model name": model_name,
        "Transform": transform,
        "Mean": cv_results.mean(),
        "Std": cv_results.std()
    }
    return msg



