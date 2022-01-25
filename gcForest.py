import numpy as np
import random
import math

from layer import layer
from k_fold_wrapper import KFoldWapper
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
from function import resample,get_logger

class gcForest(object):

    def __init__(self,config,n):

        self.estimator_configs=config["estimator_configs"]
        self.error_threshold=config["error_threshold"]
        self.resampling_rate=config["resampling_rate"]

        self.random_state=config["random_state"]
        self.max_layers=config["max_layers"]
        self.early_stop_rounds=config["early_stop_rounds"]
        self.train_evaluation=config["train_evaluation"]

        self.layers=[]
        self.LOGGER=get_logger("gcForest",n)

    def fit(self,x,y):
      
        x,n_feature=self.preprocess(x,y)
        evaluate=self.train_evaluation
        deepth=0
        x_train,y_train=x.copy(),y.copy()

        best_evaluation=0.0
        y_valid=None

        while True:
            
            y_proba=np.zeros((x_train.shape[0],len(self.estimator_configs)))
            current_layer=layer(deepth)
            self.LOGGER.info("-----------------------------------------layer-{}--------------------------------------------".format(current_layer.get_layer_id()))
            self.LOGGER.info("The shape of x_train is {}".format(x_train.shape))
            for index in range(len(self.estimator_configs)):
                config=self.estimator_configs[index].copy()
                k_fold_est=KFoldWapper(deepth,index,config,random_state=self.random_state)
                y_tmp=k_fold_est.fit(x_train,y_train,y_valid,self.error_threshold,self.resampling_rate)
                current_layer.add_est(k_fold_est)
                y_proba[:,index]+=y_tmp

            y_valid=np.mean(y_proba,axis=1)
            current_layer.weight=self.calc_weight(y_train,y_valid)
            self.layers.append(current_layer)
            self.LOGGER.info("The evaluation[{}] of layer_{} is {:.4f}, MSE={:.4f}, RMSE={:.4f}, MAE={:.4f}, pearsonr={:.4f}".format(evaluate.__name__,\
            deepth,evaluate(y_train,y_valid),mean_squared_error(y_train,y_valid),np.sqrt(mean_squared_error(y_train,y_valid)),mean_absolute_error(y_train,y_valid),pearsonr(y_train,y_valid)[0]))
            
            #  ==========x_train=============
            self.LOGGER.info("training performance")
            y_train_pred=self.predict(x)
            self.LOGGER.info("R2={:.4f}, MSE={:.4f}, RMSE={:.4f}, MAE={:.4f}, pearsonr={:.4f}".format(r2_score(y,y_train_pred),\
            mean_squared_error(y,y_train_pred),np.sqrt(mean_squared_error(y,y_train_pred)),mean_absolute_error(y,y_train_pred),pearsonr(y,y_train_pred)[0]))
            current_evaluation=evaluate(y,y_train_pred)
            current_layer.evaluation=current_evaluation

            if current_evaluation>best_evaluation:
               best_evaluation=current_evaluation
            else:
               self.layers=self.layers[0:-1]
               break

            if deepth+1>=self.max_layers:
                self.LOGGER.info("reach max layers")
                break
                
            y_proba=np.sort(y_proba,axis=1)
            x_train=np.hstack((x_train[:,0:n_feature],y_proba))

            deepth+=1
    
        self.LOGGER.info("layer_num: {}".format(len(self.layers)))



    def predict(self,x):
        weight_factors=[layer_.weight for layer_ in self.layers]
        total=np.sum(weight_factors)
        weights=[weight/total for weight in weight_factors]
        x_test=x.copy()
        n_feature=x_test.shape[1]
        y_preds=np.zeros((x_test.shape[0],len(self.layers)))
        # weight=[1.0/len(self.layer_weights) for i in range(len(self.layer_weights))]
        x_test_enhanced=None
        for i in range(len(self.layers)):
            x_test_enhanced=self.layers[i].predict(x_test)
            y_pred=self.layers[i]._predict(x_test)
            x_test=np.hstack((x_test[:,0:n_feature],x_test_enhanced))
            y_preds[:,i]+=y_pred*weights[i]
        return np.sum(y_preds,axis=1)


    def preprocess(self,x_train,y_train): 
        x_train=x_train.reshape((x_train.shape[0],-1))
        n_feature=x_train.shape[1]
        self.LOGGER.info("=========================Begin=========================")
        self.LOGGER.info("The number of samples is {}, the shape is {}".format(len(y_train),x_train[0].shape))
        self.LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        return x_train,n_feature

    
    def calc_weight(self,y_true,y_pred):
        n_sample=y_true.shape[0]
        errors=np.abs(y_true-y_pred)
        max_error=np.max(errors)
        total_error=0.0
        for i in range(n_sample):
            # ei=(y_true[i]-y_pred[i])**2/(max_error**2)
            ei=np.abs(y_true[i]-y_pred[i])/max_error
            total_error+=ei
        e=total_error/n_sample
        weight_coeff=e/(1-e)
        weight=math.log(1/weight_coeff,math.e)*weight_coeff
        return weight
