from config import DATA_PATH
from dataset import IrisDivider, IrisDataset
from trainer import Trainer
from model import MultiLayerPerceptron






if __name__ == "__main__":
    # init data
    features = ['SepalLengthCm']

    iris_ds = IrisDataset(DATA_PATH, features)
    train_dataset = IrisDataset(iris_ds.train_df)
    valid_dataset = IrisDataset(iris_ds.valid_df)
    test_dataset = IrisDataset(iris_ds.test_df)

    # init model
    input_shape = len(features)
    hidden_layers = [10]
    output_shape = 3
    model = MultiLayerPerceptron(input_shape=input_shape,
                                hidden_layers=hidden_layers,
                                output_shape=output_shape)

    # Create trainer instance
    lr = 0.001
    trainer = Trainer(model, 
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      test_dataset=test_dataset,
                      lr = lr)

    
    
