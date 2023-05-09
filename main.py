from config import DATA_PATH
from dataset import IrisDataset

if __name__ == "__main__":
    iris_ds = IrisDataset(DATA_PATH)
    print(iris_ds.df.head())
    
