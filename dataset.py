from typing import Text
import pandas as pd

class IrisDataset:
    def __init__(
        self,
        path: Text
    ) -> None:
        self.path = path
        self.df = pd.read_csv(filepath_or_buffer=self.path)
        