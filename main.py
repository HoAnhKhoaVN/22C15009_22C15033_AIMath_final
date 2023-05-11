from config import DATA_PATH
from dataset import IrisDivider, IrisDataset
from trainer import Trainer
from model import MultiLayerPerceptron
import torch


if __name__ == "__main__":
    # init data
    # 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    iris_ds = IrisDivider(DATA_PATH, features.copy())
    
    train_dataset = IrisDataset(iris_ds.df_train)
    valid_dataset = IrisDataset(iris_ds.df_valid)
    test_dataset = IrisDataset(iris_ds.df_test)

    # init model
    input_shape = len(features)
    print("input shape: ", input_shape)
    hidden_layers = [10]
    output_shape = 3
    seed = 100

    torch.manual_seed(seed)
    model = MultiLayerPerceptron(input_shape=input_shape,
                                hidden_layers=hidden_layers,
                                output_shape=output_shape)

    # Create trainer instance
    lr = 0.001
    trainer = Trainer(model, 
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      test_dataset=test_dataset,
                      learning_rate=lr,
                      batch_size=4,
                      n_epochs=50)

    trainer.train()
    
    
