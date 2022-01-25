import numpy as np
import random
import logging
import sys
import time

def get_logger(name,n):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')

    # file_handler = logging.FileHandler("logs/{}.txt".format(n))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # console_handle=logging.StreamHandler(sys.stderr)
    # console_handle.setLevel(logging.INFO)
    # console_handle.setFormatter(formatter)

    # logger.addHandler(file_handler)
    # logger.addHandler(console_handle)
    return logger

# def resample(x,y,n_new):
#     n_new=int(n_new)
#     new_x=np.zeros((n_new,x.shape[1]))
#     new_y=np.zeros((n_new))
#     sample_index=[i for i in range(x.shape[0])]
#     for i in range(n_new):
#         index=random.sample(sample_index,2)
#         sample_1,sample_2=x[index[0]],x[index[1]]
#         label_1,label_2=y[index[0]],y[index[1]]
#         r=random.random()
#         # print(r)
#         new_sample_x=r*sample_1+(1-r)*sample_2
#         new_sample_y=r*label_1+(1-r)*label_2
#         new_x[i]+=new_sample_x
#         new_y[i]+=new_sample_y
#     return new_x,new_y

def resample(x,y,n_new,n_loop=1):
    sample_index=[i for i in range(x.shape[0])]
    new_x,new_y=None,None
    for i in range(n_loop):
        new_index=random.sample(sample_index,n_new)
        new_x=x[new_index] if new_x is None else np.vstack((new_x,x[new_index]))
        new_y=y[new_index] if new_y is None else np.hstack((new_y,y[new_index]))
    return new_x,new_y

def adjust_sample(x_train,y_train,y_pred,error_threshold,resampling_rate,n_loop=1):
    
    a=1/(2*error_threshold+(1-error_threshold))
    s_r=resampling_rate*a-1

    n_train=x_train.shape[0]
    errors=np.abs(y_train-y_pred)
    rank_index=np.argsort(errors)

    hard_num=int(error_threshold*n_train)
    hard_index=rank_index[-hard_num:]
    x1,y1=x_train[hard_index],y_train[hard_index]
    x2,y2=resample(x1,y1,int(s_r*x1.shape[0]),n_loop)

    easy_num=int(a*n_train*(1-error_threshold))
    easy_index=random.sample(rank_index[0:-hard_num].tolist(),easy_num)
    x3,y3=x_train[easy_index],y_train[easy_index]

    x_train,y_train=np.vstack((x1,x2,x3)),np.hstack((y1,y2,y3))
    return x_train,y_train