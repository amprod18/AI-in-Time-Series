from torch.utils.data import Dataset

from torch import Tensor


class WindowedDataType(Dataset):
    def __init__(self, X:Tensor, y:Tensor, window_size:int):
        if (len(X) < window_size) or (window_size <= 0):
            raise ValueError("Window size should be less than the length of the dataset.")
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, index:int):
        if index >= (len(self.X) - self.window_size):
            raise IndexError
        return (self.X[index : index + self.window_size],  # Input sequence
                self.y[index + self.window_size - 1])  # Target value
        

class DatasetStorer:
    """A hierarchical container to store datasets and parameters."""
    def __init__(self, name:str, **kwargs) -> None:
        self.name = name
        for name, value in kwargs.items():
            setattr(self, name, value)
    
    def add_container(self, container_name:str, **kwargs) -> 'DatasetStorer':
        """Adds a sub-container inside the current instance."""
        setattr(self, container_name, DatasetStorer(container_name, **kwargs))
        return self
    


    