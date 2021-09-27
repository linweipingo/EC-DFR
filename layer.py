import numpy as np 

class layer(object):
    def __init__(self,layer_id):
        self.layer_id=layer_id
        self.estimators=[]
        self.evaluation=None
        self.weight=0.0
    
    def add_est(self,estimator):
        if estimator!=None:
            self.estimators.append(estimator)

    def get_layer_id(self):
        return self.layer_id

    def predict(self,x):
        values=np.zeros((x.shape[0],len(self.estimators)))
        for i in range(len(self.estimators)):
            tmp=self.estimators[i].predict(x)
            values[:,i]=tmp
        return values
    
    def _predict(self,x):
        value=None
        for est in self.estimators:
            value=est.predict(x) if value is None else value+est.predict(x)
        value/=len(self.estimators)
        return value