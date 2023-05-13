# from config import DATA_PATH
# from dataset import IrisDataset
# # from trainer import Trainer
# # from model import MultiLayerPerceptronss
# import torch


if __name__ == "__main__":
    # # init data
    # # 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
    # features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    # iris_ds = IrisDivider(DATA_PATH, features.copy())
    
    # train_dataset = IrisDataset(iris_ds.df_train)
    # valid_dataset = IrisDataset(iris_ds.df_valid)
    # test_dataset = IrisDataset(iris_ds.df_test)

    # # init model
    # input_shape = len(features)
    # print("input shape: ", input_shape)
    # hidden_layers = [1]
    # output_shape = 3
    # seed = 100

    # torch.manual_seed(seed)
    # model = MultiLayerPerceptron(input_shape=input_shape,
    #                             hidden_layers=hidden_layers,
    #                             output_shape=output_shape)
    
    # print(model)

    # # # Create trainer instance
    # # lr = 0.001
    # # trainer = Trainer(model, 
    # #                   train_dataset=train_dataset,
    # #                   valid_dataset=valid_dataset,
    # #                   test_dataset=test_dataset,
    # #                   learning_rate=lr,
    # #                   batch_size=4,
    # #                   n_epochs=50)

    # # trainer.train()

    from dataset import IrisDataset
    from config import LINK_DATA_IRIS
    import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    from method.perceptron import Perceptron

    iris_ds = IrisDataset(link = LINK_DATA_IRIS, lst_filter_feature=[1,2,3,4])
    X = np.insert(iris_ds.X_train, obj= 0, values= 1, axis= 1)
    y = iris_ds.y_train

    ppn = Perceptron(eta=0.1, n_iter=1000)

    ppn.fit(X, y)
    from sklearn.metrics import f1_score
    import numpy as np
    X_test = np.insert(iris_ds.X_test, obj= 0, values= 1, axis= 1)
    preds = [ppn.predict(X_test[idx]) for idx in range(len(X_test))]
    print(f"Top 10 kết quả dự đoán: {preds[:10]}")
    labels  = iris_ds.y_test
    print(f"Top 10 kết quả gán nhãn : {labels[:10]}")
    print(f"Độ chính xác theo f1-score: {f1_score(y_true= labels, y_pred= preds)}")
    
    
