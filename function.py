import numpy as np
import random

def generate_data(x,y,n_new):
    n_new=int(n_new)
    new_x=np.zeros((n_new,x.shape[1]))
    new_y=np.zeros((n_new))
    sample_index=[i for i in range(x.shape[0])]
    for i in range(n_new):
        index=random.sample(sample_index,2)
        # print(index)
        # index_1,index_2=index[0],index[1]
        sample_1,sample_2=x[index[0]],x[index[1]]
        label_1,label_2=y[index[0]],y[index[1]]
        r=random.random()
        # print(r)
        new_sample_x=r*sample_1+(1-r)*sample_2
        new_sample_y=r*label_1+(1-r)*label_2
        new_x[i]+=new_sample_x
        new_y[i]+=new_sample_y
    # x=np.vstack((x,new_x))
    # y=np.hstack((y,new_y))
    return new_x,new_y

def resample(x,y,n_new,n):
    sample_index=[i for i in range(x.shape[0])]
    new_x,new_y=None,None
    for i in range(n):
        new_index=random.sample(sample_index,n_new)
        new_x=x[new_index] if new_x is None else np.vstack((new_x,x[new_index]))
        new_y=y[new_index] if new_y is None else np.hstack((new_y,y[new_index]))

    return new_x,new_y