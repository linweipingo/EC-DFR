from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
from logger import get_logger



class KFoldWapper(object):
    def __init__(self,layer_id,index,config,random_state):
        self.config=config
        self.name="layer_{}, estimstor_{}, {}".format(layer_id,index,self.config["type"])
        if random_state is not None:
            self.random_state=(random_state+hash(self.name))%1000000007
        else:
            self.random_state=None
        # print(self.random_state)
        self.n_fold=self.config["n_fold"]
        self.estimators=[None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class=globals()[self.config["type"]]
        self.config.pop("type")
    
    def _init_estimator(self):
        estimator_args=self.config
        est_args=estimator_args.copy()
        est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)
    
    def fit(self,x,y):
        kf=RepeatedKFold(n_splits=self.n_fold,n_repeats=1,random_state=self.random_state)
        cv=[(t,v) for (t,v) in kf.split(x)]
        y_train_pred=np.zeros((x.shape[0],))
        for k in range(len(self.estimators)):
            est=self._init_estimator()
            train_id, val_id=cv[k]
            est.fit(x[train_id],y[train_id])
            y_pred=est.predict(x[val_id])
            y_train_pred[val_id]+=y_pred
            self.estimators[k]=est
        return y_train_pred

    def predict(self,x):
        pre_value=0
        for est in self.estimators:
            pre_value+=est.predict(x)
        pre_value/=len(self.estimators)
        return pre_value