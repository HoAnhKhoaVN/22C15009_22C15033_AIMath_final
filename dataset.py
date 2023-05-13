from typing import Text, List
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE, LINK_DATA_IRIS
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# class IrisDivider:
#     def __init__(
#         self,
#         link: Text,
#         features: List
#     ) -> None:
#         '''Head data iris
#             Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
#             0	1	5.1	3.5	1.4	0.2	Iris-setosa
#             1	2	4.9	3.0	1.4	0.2	Iris-setosa
#             2	3	4.7	3.2	1.3	0.2	Iris-setosa
#             3	4	4.6	3.1	1.5	0.2	Iris-setosa
#             4	5	5.0	3.6	1.4	0.2	Iris-setosa     
#         '''
#         self.label2int = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
#         self.link = link
#         self.df = pd.read_csv(self.link)
#         # Filter specific attribute
#         features.append('Species') 
#         self.df = self.df[features]
#         self.df['Species'] = self.df['Species'].map(self.label2int)

#         self.df_train, self.df_test, self.df_valid = self.split_train_test()

#     def split_train_test(self):
#         train_index, test_index =  train_test_split(
#             list(self.df.index),
#             test_size = TEST_SIZE,
#             random_state = RANDOM_STATE, 
#             shuffle = True
#         )
#         test_index, val_index =  train_test_split(
#             list(test_index),
#             test_size = VAL_SIZE,
#             random_state = RANDOM_STATE, 
#             shuffle = True
#         )

#         print(f"Size train have {len(train_index)} records")
#         print(f"Size test have {len(test_index)} records")
#         print(f"Size val have {len(val_index)} records")

#         df_train = self.df.filter(items=train_index, axis=0)
#         df_test = self.df.filter(items=test_index, axis=0)
#         df_valid = self.df.filter(items=val_index, axis=0)

#         return df_train, df_test, df_valid

# class IrisDataset(Dataset):
#     def __init__(self, dataframe):
#         super(IrisDataset, self).__init__()
#         self.dataframe = dataframe
    
#     def __len__(self):
#         return len(self.dataframe)
    
#     def __getitem__(self, index):
#         # Get data sample at specified index
#         row = self.dataframe.iloc[index]

#         # Load data and label from row
#         data = row.drop('Species').values.astype('float32')
#         label = row['Species']

#         # Convert data and label to PyTorch tensor
#         data = torch.tensor(data, dtype=torch.float32)
#         label = torch.tensor(label, dtype=torch.long)
#         return data, label
        


class IrisDataset:
    def __init__(
        self,
        link: Text,
        lst_filter_feature: List[int] = [1,2,3,4]
    ) -> None:
        '''Head data iris
            Id	SepalLengthCm	SepalWidthCm	PetalLengthCm	PetalWidthCm	Species
            0	1	5.1	3.5	1.4	0.2	Iris-setosa
            1	2	4.9	3.0	1.4	0.2	Iris-setosa
            2	3	4.7	3.2	1.3	0.2	Iris-setosa
            3	4	4.6	3.1	1.5	0.2	Iris-setosa
            4	5	5.0	3.6	1.4	0.2	Iris-setosa     
        '''
        self.link = link
        self.df = pd.read_csv(self.link)
        self.set_species = list(set(list(self.df['Species'])))
        # get binary dataset
        self.binary_df = self.df[self.df['Species'].isin(['Iris-setosa', 'Iris-virginica'])]

        self.train_df, self.test_df = self.split_train_test()

        self.X_train = self.train_df.iloc[:, lst_filter_feature].values

        # Set label
        # 1: Iris-setosa
        # -1: Iris-virginica
        y = self.train_df.iloc[:,5].values
        self.y_train = np.where(y == 'Iris-virginica', -1, 1)



        self.X_test = self.test_df.iloc[:, lst_filter_feature].values
        y = self.test_df.iloc[:,5].values
        self.y_test = np.where(y == 'Iris-virginica', -1, 1)



        

    def split_train_test(self):
        train_index, test_index =  train_test_split(
            list(self.binary_df.index),
            test_size = TEST_SIZE,
            random_state = RANDOM_STATE, 
            shuffle = True
        )

        # test_index, val_index =  train_test_split(
        #     list(test_index),
        #     test_size = VAL_SIZE,
        #     random_state = RANDOM_STATE, 
        #     shuffle = True
        # )

        print(f"Size train have {len(train_index)} records")
        print(f"Size test have {len(test_index)} records")
        # print(f"Size val have {len(val_index)} records")

        df_train = self.binary_df.filter(items=train_index, axis=0)
        df_test = self.binary_df.filter(items=test_index, axis=0)
        # df_val = self.binary_df.filter(items=val_index, axis=0)

        # return df_train, df_test, df_val
        return df_train, df_test, 
    

if __name__ == "__main__":
    iris_dataset = IrisDataset(link = LINK_DATA_IRIS)
    print(iris_dataset.train_df.head())