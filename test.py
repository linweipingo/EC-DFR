import numpy as np
import pickle
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.stats import pearsonr



y_vals=pickle.load(open("results_demo/y_true_1.pkl","rb"))
y_preds=pickle.load(open("results_demo/y_pred_1.pkl","rb"))

mses=[]
rmses=[]
maes=[]
r2s=[]
pearsonrs=[]

for i in range(len(y_vals)):
    print("================={}================".format(i+1))
    y_val=y_vals[i]
    y_pred=y_preds[i]
    
    mse=mean_squared_error(y_val,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_val,y_pred)
    r2=r2_score(y_val,y_pred)
    pear=pearsonr(y_val,y_pred)[0]
    # print("MSE={:.4f}".format(mse))
    # print("RMSE={:.4f}".format(rmse))
    # print("MAE={:.4f}".format(mae))
    # print("R2={:.4f}".format(r2))
    # print("pearsonr={:.4f}".format(pear))
    mses.append(mse)
    rmses.append(rmse)
    maes.append(mae)
    r2s.append(r2)
    pearsonrs.append(pear)


print("MSE mean={:.4f}, std={:.4f}".format(np.mean(mses),np.std(mses)))
print("RMSE mean={:.4f}, std={:.4f}".format(np.mean(rmses),np.std(rmses)))
print("MAE mean={:.4f}, std={:.4f}".format(np.mean(maes),np.std(maes)))
print("R^2 mean={:.4f}, std={:.4f}".format(np.mean(r2s),np.std(r2s)))
print("PEARSONR mean={:.4f}, std={:.4f}".format(np.mean(pearsonrs),np.std(pearsonrs)))
