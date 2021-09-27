import numpy as np
import random
import math

from layer import layer
from logger import get_logger
from k_fold_wrapper import KFoldWapper
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
from function import generate_data,resample

class gcForest(object):

    def __init__(self,config,n):

        self.estimator_configs=config["estimator_configs"]
        self.error_threshold=config["error_threshold"]
        self.resampling_rate=config["resampling_rate"]

        self.random_state=config["random_state"]
        self.max_layers=config["max_layers"]
        self.early_stop_rounds=config["early_stop_rounds"]
        self.if_stacking=config["if_stacking"]
        self.if_save_model=config["if_save_model"]
        self.train_evaluation=config["train_evaluation"]

        self.layers=[]
        self.LOGGER=get_logger("gcForest",n)

    def fit(self,x,y):
      
        x,n_feature=self.preprocess(x,y)
        evaluate=self.train_evaluation
        deepth=0
        x_train,y_train=x.copy(),y.copy()

        best_evaluation=999999
        best_layer=0

        while True:
            
            x_train_proba=np.zeros((x_train.shape[0],len(self.estimator_configs)))
            current_layer=layer(deepth)
            self.LOGGER.info("-----------------------------------------layer-{}--------------------------------------------".format(current_layer.get_layer_id()))
            self.LOGGER.info("The shape of x_train is {}".format(x_train.shape))
            y_train_proba=np.zeros((x_train.shape[0],))
            for index in range(len(self.estimator_configs)):
                config=self.estimator_configs[index].copy()
                k_fold_est=KFoldWapper(deepth,index,config,random_state=self.random_state)
                x_proba=k_fold_est.fit(x_train,y_train)
                current_layer.add_est(k_fold_est)
                x_train_proba[:,index]+=x_proba
                y_train_proba+=x_proba

            y_pred=y_train_proba/len(self.estimator_configs)
            current_evaluation=evaluate(y_train,y_pred)
            current_layer.evaluation=current_evaluation
            current_layer.weight=self.calc_weight(y_train,y_pred)
            self.layers.append(current_layer)
            self.LOGGER.info("The evaluation[{}] of layer_{} is {:.4f}, MSE={:.4f}, RMSE={:.4f}, MAE={:.4f}, pearsonr={:.4f}".format(evaluate.__name__,\
            deepth,current_evaluation,mean_squared_error(y_train,y_pred),np.sqrt(mean_squared_error(y_train,y_pred)),mean_absolute_error(y_train,y_pred),pearsonr(y_train,y_pred)[0]))
            
             # ==========x_train=============
            self.LOGGER.info("training performance")
            # self.LOGGER.info(y_val[0:50])
            y_train_pred=self.predict(x)
            # self.LOGGER.info(y_train_pred[0:50])
            self.LOGGER.info("R2={:.4f}, MSE={:.4f}, RMSE={:.4f}, MAE={:.4f}, pearsonr={:.4f}".format(r2_score(y,y_train_pred),\
            mean_squared_error(y,y_train_pred),np.sqrt(mean_squared_error(y,y_train_pred)),mean_absolute_error(y,y_train_pred),pearsonr(y,y_train_pred)[0]))


            # if current_evaluation<best_evaluation:
            #    best_evaluation=current_evaluation
            #    best_layer=deepth
            # else:
            #    self.layers=self.layers[0:-1]
            #    break

            if deepth+1>=self.max_layers:
                break
                
            x_train_proba=np.sort(x_train_proba,axis=1)
            x_train=np.hstack((x_train[:,0:n_feature],x_train_proba))

            """
            The errors of these samples are sorted from small to lager. According to the distribution of S-scores, 
            the error in the 70% position is selected as the threshold, dividing these sample into 2 parts, hard samples 
            and easy sample
            """
            e_r=self.error_threshold

            a=1/(2*e_r+(1-e_r))
            s_r=self.resampling_rate*a-1
        
            n_train=x_train.shape[0]
            errors=np.abs(y_train-y_pred)
            rank_indexs=np.argsort(errors)
            error_num=int(e_r*n_train)
            error_indexs=rank_indexs[-error_num:]
            x1,y1=x_train[error_indexs],y_train[error_indexs]
            index1=np.where(y1<0)[0]
            index2=np.where(y1>0)[0]
            x_error_1,x_error_2=x1[index1],x1[index2]
            y_error_1,y_error_2=y1[index1],y1[index2]
            
            x2,y2=resample(x_error_1,y_error_1,int(s_r*len(index1)),1)
            x3,y3=resample(x_error_2,y_error_2,int(s_r*len(index2)),1)
            correct_num=int(a*n_train*(1-e_r))
            correct_index=random.sample(rank_indexs[0:-error_num].tolist(),correct_num)
            x4,y4=x_train[correct_index],y_train[correct_index]

            x_train,y_train=np.vstack((x1,x2,x3,x4)),np.hstack((y1,y2,y3,y4))
            self.LOGGER.info("Original: {}, error:{}, new>0:{}, new<0:{}".format(correct_num,error_num,int(error_num/2*s_r),int(error_num/2*s_r)))
    
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
            # print(i)
            x_test_enhanced=self.layers[i].predict(x_test)
            y_pred=self.layers[i]._predict(x_test)
            if not self.if_stacking:
                x_test=np.hstack((x_test[:,0:n_feature],x_test_enhanced))
            y_preds[:,i]+=y_pred*weights[i]
        return np.sum(y_preds,axis=1)


    def preprocess(self,x_train,y_train): 
        x_train=x_train.reshape((x_train.shape[0],-1))
        n_feature=x_train.shape[1]
        self.LOGGER.info("=========================Begin=========================")
        self.LOGGER.info("The number of samples is {}, the shape is {}".format(len(y_train),x_train[0].shape))
        self.LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        self.LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking,self.if_save_model))
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
        # print(weight_coeff)
        weight=math.log(1/weight_coeff,math.e)*weight_coeff
        return weight
