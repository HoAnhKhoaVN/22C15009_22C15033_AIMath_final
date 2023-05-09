from typing import Text
import pandas as pd
from sklearn.model_selection import  train_test_split
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE, LINK_DATA_IRIS

class IrisDataset:
    def __init__(
        self,
        link: Text
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
        self.binary_df = self.df[self.df['Species'].isin(['Iris-versicolor', 'Iris-virginica'])]

        self.train_df, self.test_df, self.val_df = self.split_train_test()


        

    def split_train_test(self):
        #region 1: Split train and test index
        train_index, test_index =  train_test_split(
            list(self.binary_df.index),
            test_size = TEST_SIZE,
            random_state = RANDOM_STATE, 
            shuffle = True
        )
        #endgion

        #region 2: Split test and validation index
        test_index, val_index =  train_test_split(
            list(test_index),
            test_size = VAL_SIZE,
            random_state = RANDOM_STATE, 
            shuffle = True
        )
        #endgion

        print(f"Size train have {len(train_index)} records")
        print(f"Size test have {len(test_index)} records")
        print(f"Size val have {len(val_index)} records")

        #region 3: Filter records by index
        df_train = self.binary_df.filter(items=train_index, axis=0)
        df_test = self.binary_df.filter(items=test_index, axis=0)
        df_val = self.binary_df.filter(items=val_index, axis=0)
        #endgion

        return df_train, df_test, df_val


        

        