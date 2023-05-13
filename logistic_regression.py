from typing import Text, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE, LINK_DATA_IRIS


class IrisDivider:
    def __init__(
        self,
        link: Text,
        features: List
    ) -> None:
        '''Head data iris
            Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
            0	1	5.1	3.5	1.4	0.2	Iris-setosa
            1	2	4.9	3.0	1.4	0.2	Iris-setosa
            2	3	4.7	3.2	1.3	0.2	Iris-setosa
            3	4	4.6	3.1	1.5	0.2	Iris-setosa
            4	5	5.0	3.6	1.4	0.2	Iris-setosa     
        '''
        self.label2int = {'Iris-setosa':0, 'Iris-virginica':1}
        self.link = link
        df = pd.read_csv(self.link)
        # Filter specific attribute
        features.append('Species') 
        df = df[features]
        # Only work for binary classifier
        df = df[df['Species'].isin(["Iris-setosa", "Iris-virginica"])]
        df['Species'] = df['Species'].map(self.label2int)
        self.df = df

        self.df_train, self.df_test, self.df_valid = self.split_train_test()

    def split_train_test(self):
        train_index, test_index =  train_test_split(
            list(self.df.index),
            test_size = TEST_SIZE,
            random_state = RANDOM_STATE, 
            shuffle = True
        )
        test_index, val_index =  train_test_split(
            list(test_index),
            test_size = VAL_SIZE,
            random_state = RANDOM_STATE, 
            shuffle = True
        )

        print(f"Size train have {len(train_index)} records")
        print(f"Size test have {len(test_index)} records")
        print(f"Size val have {len(val_index)} records")

        df_train = self.df.filter(items=train_index, axis=0)
        df_test = self.df.filter(items=test_index, axis=0)
        df_valid = self.df.filter(items=val_index, axis=0)

        return df_train, df_test, df_valid

class LogisticRegression:

    def __init__(self):
        self.w = np.random.rand(5)
        self.n_iter = 100
        self.learning_rate = 0.01
        self.loss_viz = []

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def predict(self, X, y, threshold = 0.5):
        y_hat = self.sigmoid(X @ self.w)
        y_hat = np.where(y_hat > threshold, 1, 0)

        return sum(y_hat == y)/y.shape[0]
    
    def loss_func(self, y_hat, y):
        part_1 = y * np.log(y_hat)
        part_2 = (1-y) * np.log(1-y_hat)
        return -sum(part_1 + part_2) 

    def fit(self, X, y):
        w = self.w
        for i in range(self.n_iter):
            y_hat = self.sigmoid(X @ w)
            w_d = X.T @ (y_hat - y)
            w = w - self.learning_rate*w_d
            loss = self.loss_func(y_hat, y)
            print(loss)
            self.loss_viz.append(loss)
        self.w = w

    def save_loss_as_img(self):
        x = np.arange(10, len(self.loss_viz))
        y = self.loss_viz[10:]
        plt.plot(x, y)
        plt.xlabel("Iter")
        plt.ylabel("Loss") 
        plt.title("Hàm mất trong quá trình huấn luyện")
        plt.savefig("static/loss_function.png")

def convert_2_np(df): 
    y = df['Species'].to_numpy()
    X = df.drop('Species', axis=1).to_numpy()
    bias = np.ones(X.shape[0]).reshape(-1, 1)
    X = np.concatenate((bias, X), axis = 1)
    return X, y


if __name__ == "__main__":
    # Load data
    iris_divider = IrisDivider(LINK_DATA_IRIS, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm",	"PetalWidthCm"])
    df_train = iris_divider.df_train
    df_test = iris_divider.df_test

    # init model

    logistic_regression = LogisticRegression()
    # training stage
    X_train, y_train = convert_2_np(df_train)
    logistic_regression.fit(X_train, y_train) 

    # Cuối cùng, chúng ta tìm được w
    print("Trong so: [w_0, w_1, w_2, w_3, w_4] = ", logistic_regression.w)
    # testing stage
    X_test, y_test = convert_2_np(df_test)

    # Choose threshold = 0.5
    accuracy = logistic_regression.predict(X_test, y_test, 0.5)
    print("Accuracy on Testing Set: ", accuracy)

    logistic_regression.save_loss_as_img()



