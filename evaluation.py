import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

def MSE(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred)
    return mse

def RMSE(y_true,y_pred):
    return np.sqrt(MSE(y_true,y_pred))

def R2(y_true,y_pred):
    return r2_score(y_true,y_pred)