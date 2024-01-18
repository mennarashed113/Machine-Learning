import numpy as np
class LogisticRegression:

    def __init__(self,lr_rate=0.003,no_itration=1050):
        self.lr_rate=lr_rate
        self.no_iter=no_itration
        self.thetas=None
        self.node=None

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))


    def predict(self,X):
        linear_model=np.dot(X,self.thetas)+self.node
        y_predict=self.sigmoid(linear_model)
        y_classified=[1 if i>=0.5 else 0 for i in y_predict]
        return np.array(y_classified)

    def fit(self,x,y):
        n_samples,n_features=x.shape
        self.thetas=np.zeros(n_features).reshape((-1,1)) #(rows,columns)
        self.node=0
        for _ in range(self.no_iter):
            linear_model = np.dot(x, self.thetas) + self.node
            y_predict = self.sigmoid(linear_model).reshape((-1,1))
            dthetas=(1/n_samples)*np.dot(x.T,(y_predict-y))
            dnode=(1/n_samples)*np.sum(y_predict-y)
            self.thetas= self.thetas-self.lr_rate*dthetas
            self.node-=self.lr_rate*dnode