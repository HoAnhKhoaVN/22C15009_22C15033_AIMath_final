import numpy as np
import math

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def partial_tanh_2x(self, x: float):
        mau = np.exp(2*x)+ np.exp(-2*x)
        return 8/mau**2

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        # self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])

        self.errors_ = []
        N = X.shape[0]
        d = X.shape[1]
        self.w_ =np.random.randn(d,)
        print(f"Code Tang")
        for idx in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # update = self.eta * (target - self.predict(xi))
                # if self.predict(xi) != target: # misclassify
                # print("Update!!!")
                # _xi = xi.reshape(d, 1)
                # partial = -(target-self.predict(xi))*np.sqrt(self.net_input(xi)**2+ 0.0001**2)*xi
                # print(f"Đạo hàm sgn : {self.partial_tanh_2x(self.net_input(xi))}")
                wTx = self.net_input(xi)
                p1 = -(target- np.tanh(wTx))
                p2 = (1- np.tanh(wTx)**2)
                p3= xi
                partial = p1*p2*p3
                # print(f"p1: {p1} - p2 : {p2} - p3: {p3}")
                # partial = -(target- np.tanh(wTx))*(1- np.tanh(wTx)**2)*xi
                # print(f"Partial: {partial}")
                update = self.eta * partial
                self.w_ -= update
                  # self.w_[1:] += update * _xi
                  # self.w_[0] += update
                errors += 1- int((update == 0).all())
            # print(self.w_)
            # print(f"Lần {idx +1}: có {errors} điểm phân lớp sai")
            # if errors == 0.0: # Hội tự
            #     print(f"Hội tự khi với {idx + 1} lần lặp.")
            #     break
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        z = np.dot(X,self.w_)
        return z

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0, 1, -1)
    
    


if __name__== "__main__":
    # from method.perceptron import Perceptron
    from dataset import IrisDataset
    from config import LINK_DATA_IRIS
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    iris_ds = IrisDataset(link = LINK_DATA_IRIS, lst_filter_feature=[1,2])
    X = iris_ds.X_train
    y = iris_ds.y_train

    ppn = Perceptron(eta=0.1, n_iter=1000)

    ppn.fit(X, y)



