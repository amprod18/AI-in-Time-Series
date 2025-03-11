import os
import torch
import numpy as np
from tqdm import tqdm
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from deprecation import deprecated

from ..data.CustomDatatypes import WindowedDataType
from torch.utils.data import DataLoader
from torch.nn import MSELoss, functional
from torch.optim import Adam, RMSprop

from typing import *
from pandas import DataFrame, Series
from torch import Tensor
from .base import LiquidModel # TODO: should go to datatypes?
from ncps.torch import LTC, CfC


@deprecated
class LiquidModelEncapsultor:
    def __init__(self, n_neurons:int|None, model_type:LiquidModel, neuron_type:LTC|CfC, store_hidden:bool=True, verbose:int=1, seed:int=42) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_neurons = n_neurons
        self.store_hidden = store_hidden
        self.verbose = verbose
        self.seed = seed
        
        self.model_is_built = False
        self.is_trained = False
        self.losses = {'train':np.array([]), 'validation':np.array([])}
        
        self.model_type = model_type
        self.neuron_type = neuron_type

    def _with_pbar(**pbar_kwargs):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs) -> Any:
                if 'epochs' in kwargs:
                    pbar_kwargs['total'] = kwargs['epochs']
                with tqdm(dynamic_ncols=True, colour='#FF0000', **pbar_kwargs) as pbar:
                    return func(self, *args, **kwargs, pbar=pbar)
            return wrapper
        return decorator
    
    def _build_model(self) -> 'LiquidModelEncapsultor':
        if self.model_is_built:
            raise ValueError('Model is already built.')
        elif (not hasattr(self, 'n_features_in')) or (not hasattr(self, 'n_features_out')):
            raise ValueError('Model does not have information about the data to be built.')
        elif self.n_neurons is None:
            raise ValueError('Number of model neurons has not been provided. Did you forget to load a model?')
        else:
            self.model:LiquidModel = self.model_type(self.n_neurons, self.neuron_type, self.n_features_in, self.n_features_out, store_hidden=self.store_hidden, seed=self.seed)
            self.model = self.model.to(self.device)
            self.model_is_built = True
            if self.verbose > 0:
                self.model.draw_model()
        return self
    
    @staticmethod
    def _get_pbar_color(total_epochs, epoch) -> str:
        if total_epochs == 0:
            return '#FFFFFF'
        else:
            t = epoch / total_epochs
            red = int(255 * (1 - t))   # Red decreases from 255 to 0
            green = int(255 * (t))     # Green increases from 0 to 255
            return f"#{red:02X}{green:02X}00"  # Hex format: #RRGG00 (no blue component)
    
    def _create_loader(self, X:Tensor, y:Tensor, n_jobs:int=0) -> DataLoader:
        return DataLoader(WindowedDataType(X, y, self.window_size), batch_size=self.batch_size, shuffle=self.shuffle_training_data, num_workers=n_jobs, persistent_workers=n_jobs>0, pin_memory=True, prefetch_factor=2*(n_jobs>0))
    
    @_with_pbar(desc='Train loop iter', unit=' batches', postfix={'Loss':np.inf})
    def _train_loop(self, train_loader, epochs:int=1, pbar:Union[tqdm,None]=None) -> 'LiquidModelEncapsultor':
        for epoch in range(epochs):
            self.model.train() # Set in train mode
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            self.losses['train'] = np.append(self.losses['train'], loss.item())
            
            if pbar:
                pbar.colour = self._get_pbar_color(epochs, epoch)
                pbar.update(1)
                pbar.set_postfix(**{'Train Loss':f'{self.losses["train"][-1]:.4f}'})
            
        if pbar:
            pbar.colour = self._get_pbar_color(epochs, epochs)
        return self
     
    def fit(self, X_train:DataFrame, y_train:DataFrame, epochs:int, shuffle_training_data:bool=False, learning_rate:Union[int,float]=0.01, n_jobs:int=8, window_size:int=1, batch_size:int=1, optimizer:Union[Adam]=Adam, loss:Union[MSELoss]=MSELoss) -> 'LiquidModelEncapsultor':
        self.features = X_train.columns
        self.n_features_in:int = X_train.shape[-1]
        self.n_features_out:int = y_train.shape[-1]
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_training_data = shuffle_training_data
        self.learning_rate = learning_rate
        self.y_train = y_train
        self.n_jobs = n_jobs
        
        train_loader = self._create_loader(Tensor(X_train.values), Tensor(y_train.values), n_jobs=n_jobs)
        
        self._build_model()
        
        if not self.is_trained:
            self.losses['train'] = np.array([])
        

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate) # Optimizer
        self.loss_function = loss() # Loss function
        
        self._train_loop(train_loader, epochs=self.epochs)

        self.is_trained = True
        return self

    @_with_pbar(desc='Batches to predict', unit=' batches')
    def predict(self, X_test:Tensor, pbar:Union[tqdm,None]=None) -> Tensor:
        self.model.eval()
        self.model.store_hidden = False
        self.shuffle_training_data = False
        test_loader = self._create_loader(Tensor(X_test.values), Tensor(X_test.values)[:, 0], n_jobs=self.n_jobs)
                
        with torch.no_grad():
            predictions = []
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                predictions.append(self.model(batch).cpu().detach())
        return np.concatenate(predictions, axis=0)
    
    def score(self, X_test:Tensor, y_test:Tensor) -> float:
        # y_test = Tensor(y_test.values).to(self.device)
        self.model.store_hidden = False
        predictions = self.predict(X_test)
        return functional.mse_loss(Tensor(predictions), Tensor(y_test.values[self.window_size:]).flatten()).item()
    
    def plot_losses(self) -> 'LiquidModelEncapsultor':
        assert self.is_trained, 'Model is not trained, train the model first to get some loss values.'
        
        sns.set_style('darkgrid')
        sns.lineplot(x=range(1, self.epochs+1), y=self.losses['train'], label='train')
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel(f'{self.loss_function._get_name()} Loss')
        if self.losses['validation']:
            sns.lineplot(x=range(1, self.epochs+1), y=self.losses['validation'], label='validation')
            plt.title('Train Losses')
            plt.legend()
        plt.show()
        return self
    
    def explain(self, y_train:Union[DataFrame,Series], X_train:Union[DataFrame,None]=None) -> 'LiquidModelEncapsultor':
        assert self.is_trained, 'Model has not yet been trained. Train the model to generate explanations.'
        self.model.explain(self.y_train, self.window_size, X_train=X_train)
        return self
    
    def _set_up_parameters(self, metadata:dict[str, object|int|float]) -> 'LiquidModelEncapsultor':
        for param, value in metadata.items():
            self.__setattr__(param, value)
        return self

    def save_model(self, model_path:Path) -> 'LiquidModelEncapsultor':
        metadata = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}
        if not os.path.exists(model_path.parent):
            os.makedirs(model_path.parent)
        torch.save({'model_state_dict':self.model.state_dict(), 'metadata':metadata}, model_path)
        return self

    def load_model(self, model_path) -> 'LiquidModelEncapsultor':
        checkpoint = torch.load(model_path)
        self._set_up_parameters(checkpoint['metadata'])
        self.model_is_built = False
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self
    
    
class TorchSklearnifier:
    def __init__(self, model, njobs:int=0, verbose:int=1, seed:int=42, **model_kwargs) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_type = model
        
        self.model_kwargs = model_kwargs
        self.verbose = verbose
        self.seed = seed
        self.njobs = njobs
        
        self.model_is_built = False
        self.is_trained = False
        self.losses = {'train':np.array([]), 'validation':np.array([])}

    def _with_pbar(**pbar_kwargs):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs) -> Any:
                if 'epochs' in kwargs:
                    pbar_kwargs['total'] = kwargs['epochs']
                with tqdm(dynamic_ncols=True, colour='#FF0000', **pbar_kwargs) as pbar:
                    return func(self, *args, **kwargs, pbar=pbar)
            return wrapper
        return decorator
    
    def _build_model(self) -> 'TorchSklearnifier':
        if self.model_is_built:
            raise ValueError('Model is already built.')
        elif (not hasattr(self, 'n_features_in')) or (not hasattr(self, 'n_features_out')):
            raise ValueError('Model does not have information about the data to be built.')
        else:
            # TODO: Oopsie, how tf do I generalize?
            self.model = self.model_type(**self.model_kwargs)
            self.model = self.model.to(self.device)
            self.model_is_built = True
            # if self.verbose > 0:
            #     self.model.summary()
        return self
    
    @staticmethod
    def _get_pbar_color(total_epochs:int, epoch:int) -> str:
        if total_epochs == 0:
            return '#FFFFFF'
        else:
            t = epoch / total_epochs
            red = int(255 * (1 - t))   # Red decreases from 255 to 0
            green = int(255 * (t))     # Green increases from 0 to 255
            return f"#{red:02X}{green:02X}00"  # Hex format: #RRGG00 (no blue component)
    
    def _create_loader(self, X:Tensor, y:Tensor) -> DataLoader:
        return DataLoader(WindowedDataType(X, y, self.window_size), batch_size=self.batch_size, shuffle=self.shuffle_training_data, num_workers=self.njobs, persistent_workers=self.njobs>0, pin_memory=True, prefetch_factor=2*(self.njobs>0))
    
    # TODO: Add the possibility to use a validation dataloader
    @_with_pbar(desc='Train loop iter', unit=' batches', postfix={'Loss':np.inf})
    def _train_loop(self, train_loader:DataLoader, epochs:int=1, pbar:Union[tqdm,None]=None) -> 'TorchSklearnifier':
        for epoch in range(epochs):
            self.model.train() # Set in train mode
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.loss_function(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            self.losses['train'] = np.append(self.losses['train'], loss.item())
            if pbar:
                pbar.colour = self._get_pbar_color(epochs, epoch)
                pbar.update(1)
                pbar.set_postfix(**{'Train Loss':f'{self.losses["train"][-1]:.4f}'})
            
        if pbar:
            pbar.colour = self._get_pbar_color(epochs, epochs)
        return self
     
    def fit(self, X_train:DataFrame, y_train:DataFrame, epochs:int=1, window_size:int=1, batch_size:int=1, learning_rate:Union[int,float]=0.01, shuffle_training_data:bool=False, optimizer:Union[Adam,RMSprop]=Adam, loss:Union[MSELoss]=MSELoss) -> 'TorchSklearnifier':
        self.features = X_train.columns
        self.n_features_in:int = X_train.shape[-1]
        self.n_features_out:int = y_train.shape[-1]
        self.model_kwargs['in_features'] = self.n_features_in
        self.model_kwargs['out_features'] = self.n_features_out
        
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_training_data = shuffle_training_data
        self.learning_rate = learning_rate
        
        train_loader = self._create_loader(Tensor(X_train.values), Tensor(y_train.values))
        
        self._build_model()
        
        if not self.is_trained:
            self.losses['train'] = np.array([])
            self.losses['validation'] = np.array([])
        

        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate) # Optimizer
        self.loss_function = loss() # Loss function
        
        self._train_loop(train_loader, epochs=self.epochs)

        self.is_trained = True
        return self

    @_with_pbar(desc='Batches to predict', unit=' batches')
    def predict(self, X_test:DataFrame, pbar:Union[tqdm,None]=None) -> Tensor:
        self.model.eval()
        self.shuffle_training_data = False
        test_loader = self._create_loader(Tensor(X_test.values), Tensor(X_test.values)[:, 0])
                
        with torch.no_grad():
            predictions = []
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                predictions.append(self.model(batch).cpu().detach())
        return Tensor(np.concatenate(predictions, axis=0))
    
    def score(self, X_test:DataFrame, y_test:DataFrame) -> float:
        self.model.store_hidden = False
        predictions = self.predict(X_test)
        return self.loss_function(Tensor(predictions), Tensor(y_test.values[self.window_size:]).flatten()).item()
    
    def plot_losses(self) -> 'TorchSklearnifier':
        assert self.is_trained, 'Model is not trained, train the model first to get some loss values.'
        
        sns.set_style('darkgrid')
        sns.lineplot(x=range(1, self.epochs+1), y=self.losses['train'], label='train')
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel(f'{self.loss_function._get_name()} Loss')
        if self.losses['validation'].size > 0:
            sns.lineplot(x=range(1, self.epochs+1), y=self.losses['validation'], label='validation')
            plt.title('Validation Losses')
            plt.legend()
        plt.show()
        return self
    
    # TODO: Attempt to generalize for every pytorch model
    def explain(self, y_train:Union[DataFrame,Series], X_train:Union[DataFrame,None]=None) -> 'TorchSklearnifier':
        assert self.is_trained, 'Model has not yet been trained. Train the model to generate explanations.'
        self.model.explain(y_train, self.window_size, X_train=X_train)
        return self
    
    def _set_up_parameters(self, metadata:Dict[str,Union[object,int,float]]) -> 'TorchSklearnifier':
        for param, value in metadata.items():
            self.__setattr__(param, value)
        return self

    def save_model(self, model_path:Path) -> 'TorchSklearnifier':
        metadata = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)}
        if not os.path.exists(model_path.parent):
            os.makedirs(model_path.parent)
        torch.save({'model_state_dict':self.model.state_dict(), 'metadata':metadata}, model_path)
        return self

    def load_model(self, model_path:Path) -> 'TorchSklearnifier':
        checkpoint = torch.load(model_path)
        self._set_up_parameters(checkpoint['metadata'])
        self.model_is_built = False
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self