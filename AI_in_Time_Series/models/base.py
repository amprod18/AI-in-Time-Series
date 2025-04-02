from torch.nn import AdaptiveAvgPool1d, Linear, Module, Sequential, LSTM, ReLU, Softplus, Identity, Parameter
from ncps.wirings import AutoNCP, FullyConnected
from ncps.torch import LTC, CfC, LTCCell, CfCCell
import torch
from torchdiffeq import odeint

from .odes import ODEFunc

from sklearn.decomposition import PCA
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from deprecation import deprecated

from torch import Tensor
from typing import *
from pandera.typing import DataFrame, Series
from ncps.wirings import Wiring, FullyConnected


class BaselineModel():
    def __init__(self, time_indexer:str, features:Union[str, list[str]], look_ahead:int=1, strategy:str='copycat') -> None:
        self.look_ahead = look_ahead
        self._time_indexer = time_indexer
        self.features = features if isinstance(features, list) else [features]
        self.strategy = strategy.lower()
        self.metrics = None
        
        self.valid_strategies = ["copycat", "naive_seasonal", "moving_average", "linear"]
        if strategy not in self.valid_strategies:
            raise ValueError(f"Strategy must be one of {self.valid_strategies}, founc {strategy}")
        
    @staticmethod
    def get_metrics(y_true:Union[DataFrame,Series], y_pred:Union[DataFrame,Series]) -> DataFrame:
        metrics = {}
        for target in y_true.columns:
            mse = mean_squared_error(y_true[target], y_pred[target])
            mae = mean_absolute_error(y_true[target], y_pred[target])
            mape = mean_absolute_percentage_error(y_true[target], y_pred[target])
            r2 = r2_score(y_true, y_pred)
            metrics[f'{target}_mse'] = mse
            metrics[f'{target}_mae'] = mae
            metrics[f'{target}_mape'] = mape
            metrics[f'{target}_r2'] = r2
                        
        return pd.DataFrame.from_dict({'Metrics':metrics}, orient='index')
        
    def fit(self, X:DataFrame, y:Union[DataFrame,Series,None]=None, window_size:int=1, seasonal_period:Union[int,None]=None) -> 'BaselineModel':
        assert len(X) > 1, 'Two samples are required to determine the time sample rate.'
        self.window_size = window_size
        self.targets = list(y.columns)
        self.time_sample_rate = X[self._time_indexer].iloc[1] - X[self._time_indexer].iloc[0]
        self.X_train = X
        self.y_train = y
        if self.strategy == "naive_seasonal" and seasonal_period is None:
            raise ValueError("seasonal_period must be provided for naive_seasonal strategy")
        self.seasonal_period = seasonal_period
        print(f'Sample rate detected: {self.time_sample_rate}')
        return self
        
    def predict(self, X:DataFrame, y:Union[DataFrame,Series,None]=None) -> Union[DataFrame,Series]:
        time_indexer = X[self._time_indexer]
        if self.strategy == "copycat":
            post = X[self.features].iloc[-1:]
            post.loc[:, self._time_indexer] = X[self._time_indexer].values[-1] + self.time_sample_rate
            post.index += 1
            post = post[X.columns]
            y_pred = pd.concat([pd.concat([time_indexer, X[self.features].shift(self.look_ahead).iloc[self.window_size:]], axis=1).dropna(), post], axis=0).iloc[1:]
        elif self.strategy == "naive_seasonal":
            y_pred = pd.concat([time_indexer, X[self.features].shift(self.seasonal_period).iloc[self.window_size:]], axis=1).dropna()
        elif self.strategy == "moving_average":
            y_pred = pd.concat([time_indexer, X[self.features].rolling(self.window_size).mean().shift(self.look_ahead).iloc[self.window_size:]], axis=1).dropna()
        elif self.strategy == "linear":
            y_pred = self._linear_extrapolation(X, time_indexer)
            
        y_pred.index += -1
        y_pred = y_pred.rename(columns=dict(zip(self.features, self.targets)))
        return y_pred
    
    def _linear_extrapolation(self, X:DataFrame, time_indexer:Series) -> DataFrame:
        # Simple linear fit on last window_size points, extrapolate one step
        y_pred = X[self.features].copy()
        for feature in self.features:
            window = X[feature].iloc[-self.window_size:]
            x = np.arange(len(window))
            coeffs = np.polyfit(x, window, 1)  # Linear fit
            next_val = coeffs[0] * len(window) + coeffs[1]
            y_pred[feature] = np.append(X[feature].values[self.window_size:], next_val)
        return pd.concat([time_indexer[self.window_size:], y_pred[self.window_size:]], axis=1)
    
    def evaluate(self, X:DataFrame, y:Union[DataFrame,Series,None], plot_prediction:bool=True) -> DataFrame:
        y = pd.concat([X[self._time_indexer], y], axis=1).iloc[self.window_size:]
        y_pred = self.predict(X)
        
        if plot_prediction:
            for target in self.targets:
                sns.lineplot(data=y, x=self._time_indexer, y=target, sort=False, label=f'{target}')
                sns.lineplot(data=y_pred, x=self._time_indexer, y=target, sort=False, label=f'{target}_pred')
            
        not_nan_mask = y[self.targets[0]].notna()
        self.metrics = self.get_metrics(y[not_nan_mask][self.targets], y_pred[not_nan_mask][self.targets])
        self.metrics.index = ['baseline']
        return self.metrics
    
    def compare_models(self, *metrics:DataFrame, names:Union[list[str],None]=None, X:Union[DataFrame,None]=None, y:Union[DataFrame,Series,None]=None) -> DataFrame:
        if self.metrics is not None:
            assert all([model.columns.equals(self.metrics.columns) for model in metrics]), 'Metrics must be the same for comparison. If unsure us the BaselineModel.get_metrics(y_true, y_pred) for easy construction.'
            if names is None:
                names = range(1, len(metrics)+1)
            elif len(names) < len(metrics):
                names.extend(range(len(names)+1, len(metrics)+1))
            
            models_metrics = pd.concat(metrics, axis=0)
            models_metrics.index = [f'Performance for model {name}' for name in names][:len(metrics)]
            comparison = models_metrics/self.metrics.iloc[0]
            return pd.concat([self.metrics, comparison], axis=0)
        
        elif (X is None) or (y is None):
            raise ValueError('Provide a valid X and y input to evaluate the baseline model.')
        else:
            self.evaluate(X, y)
            self.compare_model(metrics)
 
            
# TODO: Review
class TimeSeriesModel(Module):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()
        self.hidden = []
        self.store_hidden = True

    def _reset_hidden_states(self):
        self.hidden = []
        return self

    def _fit_explainer(self):
        if not self.hidden:
            raise ValueError("No hidden states stored. Set store_hidden=True and run forward pass.")
        hidden_states = np.concatenate(self.hidden, axis=0)
        self.explainer = PCA(n_components=2)
        self.explainer.fit(hidden_states)
        return self

    def explain(self, y_train:Union[DataFrame,Series], window_size:int=1, X_train:Union[DataFrame,None]=None):
        self._fit_explainer()
        pca_components = self.explainer.transform(self.hidden[-len(y_train) + window_size:])
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.subplots_adjust(wspace=0.01)
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=pca_components[:, 0], ax=ax[0])
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=pca_components[:, 1], ax=ax[1])
        ax[0].set_ylabel("y")
        ax[0].set_title("First PCA component")
        ax[1].set_title("Second PCA component")
        plt.show()
        print(f"Variance Explained: {100 * self.explainer.explained_variance_ratio_.sum():.2f}%")
        return self

    def summary(self):
        raise NotImplementedError("Subclasses must implement summary()")


class LSTMModel(Module):
    def __init__(self, in_features:int, hidden_size:int, out_features:int, num_layers:int=1) -> None:
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(in_features, hidden_size, num_layers, batch_first=True)
        self.model = Sequential(ReLU(), Linear(hidden_size, out_features))
        
    def forward(self, X:Tensor) -> Tensor:
        x, _ = self.lstm(X)
        x = self.model(x)
        return x[:, -1, :]


class NeuralODEModel(Module):
    def __init__(self, func:ODEFunc, in_features:int, out_features:int, hidden_size:int, solver:str='rk4') -> None:
        super(NeuralODEModel, self).__init__()
        self.func = func(in_features-1, hidden_size, hidden_size)
        self.solver = solver
        self.linear_out = Linear(hidden_size, out_features)
    
    def forward(self, x:Tensor) -> Tensor:
        t = x[:, :, 0]
        y0 = x[:, -1, 1:]
        solution = odeint(self.func, y0, t[0], method=self.solver)
        return self.linear_out(solution[-1])


class GeneralizedNeuralODE(Module):
    def __init__(self, in_features, hidden_size, out_features, odefunc:ODEFunc, solver='dopri5', store_hidden:bool=True, seed:int=42) -> None:
        super(GeneralizedNeuralODE, self).__init__()
        
        self.input_layer = Linear(in_features-1, hidden_size)
        self.ode_func = odefunc(hidden_size, hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, out_features)
        self.solver = solver
        
        self.store_hidden = store_hidden
        self._reset_hidden_states()
        
    def _reset_hidden_states(self) -> None:
        self.hidden_input = []
        self.hidden_lnn = []
    
    def forward(self, x) -> Tensor:
        batch_size, window_size, n_features = x.shape
        t_span = x[:, :, 0]  # Extract time column
        x = x[:, :, 1:]  # Remove time column from input
        x = x.view(-1, n_features - 1)  # Flatten batch and window size
        x = self.input_layer(x)
        
        # Solve ODE for each sample individually due to different t_span per batch
        outputs = []
        for i in range(batch_size):
            t_sorted, indices = torch.sort(t_span[i])  # Ensure monotonic time
            x_i = odeint(self.ode_func, x[i * window_size:(i + 1) * window_size], t_sorted, method=self.solver, rtol=1e-5, atol=1e-7)[-1] # Only keeping the last value of the window, maybe the other should be kept as hidden states
            outputs.append(x_i[torch.argsort(indices)])  # Restore original order
        
        x = torch.stack(outputs).view(batch_size, window_size, -1)
        x = self.output_layer(x)[:, -1, :]
        
        """
        if self.store_hidden:
            self.hidden_input.append(hidden_input.cpu().detach().numpy())
            self.hidden_lnn.append(hidden_lnn.cpu().detach().numpy())
            
        return self.out_layer(x)[:, -1, :]
        """
        
        return x

    def summary(self) -> None:
        sns.set_style("white")
        
        plt.figure(figsize=(6, 5))
        plt.title(f"Input NCP Layer Architecture with LTC Neurons")
        legend_handles = self.ncp_wiring_input.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(6, 5))
        plt.title(f"NCP Layer Architecture")
        legend_handles = self.ncp_wiring_lnn.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    def _fit_explainer(self) -> 'LiquidNeuralNetwork':
        self.hidden_input_states = np.concatenate(self.hidden_input, axis=0)
        self.hidden_lnn_states = np.concatenate(self.hidden_lnn, axis=0)
        
        self.input_explainer = PCA(n_components=2)
        self.input_explainer.fit(self.hidden_input_states)
        
        self.lnn_explainer = PCA(n_components=2)
        self.lnn_explainer.fit(self.hidden_lnn_states)
        return self
    
    def explain(self, y_train:Union[DataFrame,Series], window_size:int=1, X_train:Union[DataFrame,None]=None) -> 'LiquidNeuralNetwork':
        self._fit_explainer()
        lnn_pca_components = self.lnn_explainer.transform(self.hidden_lnn_states[-len(y_train)+window_size:])
        input_pca_components = self.input_explainer.transform(self.hidden_input_states[-len(y_train)+window_size+1:])

        if X_train is not None:
            fig, ax = plt.subplots(nrows=2, ncols=len(X_train.columns), figsize=(6*len(X_train.columns), 12), sharey=True)
            fig.subplots_adjust(wspace=0.01, hspace=0.05)
            X_train_ = X_train.iloc[window_size+1:] # TODO: Fix that 1, which refers to the steps into the future

            for i, col in enumerate(X_train.columns):
                sns.scatterplot(x=X_train_[col].index, y=X_train_[col].values.flatten(), c=input_pca_components[:, 0], ax=ax[0, i])
                sns.scatterplot(x=X_train_[col].index, y=X_train_[col].values.flatten(), c=input_pca_components[:, 1], ax=ax[1, i])
                ax[0, i].set_xlabel(None)
                ax[0, i].set_title(col)
            ax[0, 0].set_ylabel('First PCA component of neuron activations')
            ax[1, 0].set_ylabel('Second PCA component of neuron activations')
            plt.show()
        
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.subplots_adjust(wspace=0.01)
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=lnn_pca_components[:, 0], ax=ax[0])
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=lnn_pca_components[:, 1], ax=ax[1])
        ax[0].set_ylabel('y')
        ax[0].set_title('First PCA component of neuron activations')
        ax[1].set_title('Second PCA component of neuron activations')
        ax[0].xaxis.set_tick_params(rotation=90)
        ax[1].xaxis.set_tick_params(rotation=90)
        plt.show()
        print(f'Variance Explained: {100*self.lnn_explainer.explained_variance_ratio_.sum():.2f}%')
        return self


# TODO: To be done yet
class LiquidModel(Module):
    def __init__(self, n_neurons:int, neuron_type:LTC|CfC, in_features:int, out_features:int, store_hidden:bool=True, seed:int=42) -> None:
        ...
        
    def summary(self) -> None:
        ...
        
    def forward(self, x) -> Tensor:
        return x


class LiquidNeuralNetwork(Module):
    def __init__(self, n_neurons:int, neuron_type:LTC|CfC, in_features:int, out_features:int, store_hidden:bool=True, seed:int=42) -> None:
        super(LiquidNeuralNetwork, self).__init__()
        
        self.ncp_wiring_output:AutoNCP = AutoNCP(n_neurons, out_features, seed=seed) # LNN
        self.ncp_layer_output:LTC|CfC = neuron_type(in_features, self.ncp_wiring_output, batch_first=True)
        self.out_layer = Sequential(AdaptiveAvgPool1d(out_features), Linear(out_features, out_features))
        
        self.store_hidden = store_hidden
        self._reset_hidden_states()

    def _reset_hidden_states(self) -> 'LiquidNeuralNetwork':
        self.hidden = []
        return self
        
    def forward(self, x) -> Tensor:
        x, hidden = self.ncp_layer_output.forward(x)
        if self.store_hidden:
            self.hidden.append(hidden.cpu().detach().numpy())
        return self.out_layer.forward(x)[:, -1, :]
    
    def summary(self) -> 'LiquidNeuralNetwork':
        sns.set_style("white")
        plt.figure(figsize=(6, 5))
        plt.title(f"NCP Layer Architecture")
        legend_handles = self.ncp_wiring_output.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        return self
    
    def _fit_explainer(self) -> 'LiquidNeuralNetwork':
        self.hidden_states = np.concatenate(self.hidden, axis=0)
        self.explainer = PCA(n_components=2)
        self.explainer.fit(self.hidden_states)
        return self
    
    def explain(self, y_train:Union[DataFrame,Series], window_size:int=1, X_train:Union[DataFrame,None]=None) -> 'LiquidNeuralNetwork':
        self._fit_explainer()
        pca_components = self.explainer.transform(self.hidden_states[-len(y_train)+window_size:])

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.subplots_adjust(wspace=0.01)
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=pca_components[:, 0], ax=ax[0])
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=pca_components[:, 1], ax=ax[1])
        ax[0].set_ylabel('y')
        ax[0].set_title('First PCA component of neuron activations')
        ax[1].set_title('Second PCA component of neuron activations')
        ax[0].xaxis.set_tick_params(rotation=90)
        ax[1].xaxis.set_tick_params(rotation=90)
        plt.show()
        print(f'Variance Explained: {100*self.explainer.explained_variance_ratio_.sum():.2f}%')
        return self



@deprecated
class _GeneralizedLiquidNeuralNetwork_old(Module):
    def __init__(self, n_neurons:int, neuron_type:LTC|CfC, in_features:int, out_features:int, store_hidden:bool=True, seed:int=42) -> None:
        super(_GeneralizedLiquidNeuralNetwork_old, self).__init__()
        
        self.ncp_wiring_input:FullyConnected = FullyConnected(in_features, None, erev_init_seed=seed) # GLNN
        self.ncp_wiring_lnn:AutoNCP = AutoNCP(n_neurons, out_features, seed=seed) # LNN
        self.ncp_layer_input:LTC|CfC = neuron_type(in_features, self.ncp_wiring_input, batch_first=True)
        self.ncp_layer_lnn:LTC|CfC = neuron_type(in_features, self.ncp_wiring_lnn, batch_first=True)
        self.out_layer = Sequential(AdaptiveAvgPool1d(out_features), Linear(out_features, out_features))
        
        self.store_hidden = store_hidden
        self._reset_hidden_states()
        
    def _reset_hidden_states(self) -> None:
        self.hidden_input = []
        self.hidden_lnn = []
        
    def forward(self, x) -> Tensor:
        x, hidden_input = self.ncp_layer_input.forward(x)
        x, hidden_lnn = self.ncp_layer_lnn.forward(x)

        if self.store_hidden:
            self.hidden_input.append(hidden_input.cpu().detach().numpy())
            self.hidden_lnn.append(hidden_lnn.cpu().detach().numpy())
            
        return self.out_layer(x)[:, -1, :]
    
    def summary(self) -> None:
        sns.set_style("white")
        
        plt.figure(figsize=(6, 5))
        plt.title(f"Input NCP Layer Architecture with LTC Neurons")
        legend_handles = self.ncp_wiring_input.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(6, 5))
        plt.title(f"NCP Layer Architecture")
        legend_handles = self.ncp_wiring_lnn.draw_graph(layout='circular', draw_labels=False, neuron_colors={"command": "tab:cyan"})
        plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    def _fit_explainer(self) -> 'LiquidNeuralNetwork':
        self.hidden_input_states = np.concatenate(self.hidden_input, axis=0)
        self.hidden_lnn_states = np.concatenate(self.hidden_lnn, axis=0)
        
        self.input_explainer = PCA(n_components=2)
        self.input_explainer.fit(self.hidden_input_states)
        
        self.lnn_explainer = PCA(n_components=2)
        self.lnn_explainer.fit(self.hidden_lnn_states)
        return self
    
    def explain(self, y_train:Union[DataFrame,Series], window_size:int=1, X_train:Union[DataFrame,None]=None) -> 'LiquidNeuralNetwork':
        self._fit_explainer()
        lnn_pca_components = self.lnn_explainer.transform(self.hidden_lnn_states[-len(y_train)+window_size:])
        input_pca_components = self.input_explainer.transform(self.hidden_input_states[-len(y_train)+window_size+1:])

        if X_train is not None:
            fig, ax = plt.subplots(nrows=2, ncols=len(X_train.columns), figsize=(6*len(X_train.columns), 12), sharey=True)
            fig.subplots_adjust(wspace=0.01, hspace=0.05)
            X_train_ = X_train.iloc[window_size+1:] # TODO: Fix that 1, which refers to the steps into the future

            for i, col in enumerate(X_train.columns):
                sns.scatterplot(x=X_train_[col].index, y=X_train_[col].values.flatten(), c=input_pca_components[:, 0], ax=ax[0, i])
                sns.scatterplot(x=X_train_[col].index, y=X_train_[col].values.flatten(), c=input_pca_components[:, 1], ax=ax[1, i])
                ax[0, i].set_xlabel(None)
                ax[0, i].set_title(col)
            ax[0, 0].set_ylabel('First PCA component of neuron activations')
            ax[1, 0].set_ylabel('Second PCA component of neuron activations')
            plt.show()
        
        fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.subplots_adjust(wspace=0.01)
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=lnn_pca_components[:, 0], ax=ax[0])
        sns.scatterplot(x=y_train.index[window_size:], y=y_train.iloc[window_size:].values.flatten(), c=lnn_pca_components[:, 1], ax=ax[1])
        ax[0].set_ylabel('y')
        ax[0].set_title('First PCA component of neuron activations')
        ax[1].set_title('Second PCA component of neuron activations')
        ax[0].xaxis.set_tick_params(rotation=90)
        ax[1].xaxis.set_tick_params(rotation=90)
        plt.show()
        print(f'Variance Explained: {100*self.lnn_explainer.explained_variance_ratio_.sum():.2f}%')
        return self
    

class LTCCellCustom(LTCCell):
    def __init__(self, wiring:Wiring, in_features:Union[int,None]=None, input_mapping:str="affine", output_mapping:str="affine", ode_solver:str="dopri5", 
                 rtol:float=1e-5, atol:float=1e-7, epsilon:float=1e-8, implicit_param_constraints:bool=False):
        """A custom Liquid Time-Constant (LTC) cell using torchdiffeq.odeint.

        Args:
            wiring: Wiring configuration (from ncps.wirings)
            in_features: Number of input features
            input_mapping: Input transformation ("affine", "linear", or None)
            output_mapping: Output transformation ("affine", "linear", or None)
            ode_solver: ODE solver method (e.g., "dopri5", "euler", "rk4")
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            epsilon: Small value to avoid division by zero
            implicit_param_constraints: Whether to enforce constraints implicitly
        """
        super(LTCCellCustom, self).__init__(wiring=wiring, in_features=in_features, input_mapping=input_mapping, output_mapping=output_mapping, 
                                            ode_unfolds=1, epsilon=epsilon, implicit_param_constraints=implicit_param_constraints)
        
        self.ode_solver = ode_solver
        self.rtol = rtol
        self.atol = atol

    def _ode_dynamics(self, t, state, inputs, sensory_w_activation, sensory_rev_activation):
        """Compute dv/dt for the LTC dynamics.

        Args:
            t: Time point (scalar, required by odeint)
            state: Current state v (tensor of shape [batch, state_size])
            inputs: Input tensor (pre-mapped)
            sensory_w_activation: Precomputed sensory weights
            sensory_rev_activation: Precomputed sensory reversal potentials
        """
        v_pre = state

        # Neuron-neuron interactions
        w_activation = self.make_positive_fn(self._params["w"]) * self._sigmoid(
            v_pre, self._params["mu"], self._params["sigma"]
        )
        w_activation = w_activation * self._params["sparsity_mask"]
        rev_activation = w_activation * self._params["erev"]

        # Sum contributions
        w_numerator = torch.sum(rev_activation, dim=1) + torch.sum(sensory_rev_activation, dim=1)
        w_denominator = torch.sum(w_activation, dim=1) + torch.sum(sensory_w_activation, dim=1)

        gleak = self.make_positive_fn(self._params["gleak"])
        cm = self.make_positive_fn(self._params["cm"])

        # dv/dt = (I - g_leak * (v - v_leak)) / C_m
        dv_dt = (
            -gleak * (v_pre - self._params["vleak"]) + w_numerator - w_denominator * v_pre
        ) / (cm + self._epsilon)

        return dv_dt
    
    def forward(self, inputs, states, elapsed_time=1.0):
        """Forward pass using torchdiffeq.odeint.

        Args:
            inputs: Input tensor [batch, sensory_size]
            states: Initial state tensor [batch, state_size]
            elapsed_time: Time span to integrate over (default 1.0)
        Returns:
            outputs: Output tensor [batch, motor_size]
            next_state: Next state tensor [batch, state_size]
        """
        # Map inputs
        inputs = self._map_inputs(inputs)

        # Precompute sensory contributions (constant w.r.t. state)
        sensory_w_activation = self.make_positive_fn(
            self._params["sensory_w"]
        ) * self._sigmoid(
            inputs, self._params["sensory_mu"], self._params["sensory_sigma"]
        )
        sensory_w_activation = sensory_w_activation * self._params["sensory_sparsity_mask"]
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Define the ODE function with inputs fixed
        def ode_func(t, state):
            return self._ode_dynamics(t, state, inputs, sensory_w_activation, sensory_rev_activation)

        # Time points to solve over
        t = torch.tensor([0.0, elapsed_time], device=inputs.device, dtype=torch.float32)

        # Solve ODE
        state_trajectory = odeint(
            ode_func,
            states,
            t,
            method=self.ode_solver,
            rtol=self.rtol,
            atol=self.atol,
        )

        # state_trajectory shape: [time_points, batch, state_size]
        next_state = state_trajectory[-1]  # Take the final state

        # Map outputs
        outputs = self._map_outputs(next_state)

        # Apply constraints if needed
        self.apply_weight_constraints()

        return outputs, next_state


class LTCCustom(LTC):
    def __init__(self, input_size:int, units:Union[Wiring,int], return_sequences:bool=True, batch_first:bool=True, mixed_memory:bool=False, input_mapping:str="affine", output_mapping:str="affine", 
                 ode_unfolds:int=6, ode_solver="dopri5", rtol=1e-5, atol=1e-7, epsilon=1e-8, implicit_param_constraints=True):
        super(LTCCustom, self).__init__(input_size, units, return_sequences=return_sequences, batch_first=batch_first, mixed_memory=mixed_memory, input_mapping=input_mapping, output_mapping=output_mapping, ode_unfolds=ode_unfolds)

        if isinstance(units, Wiring):
            wiring = units
        else:
            wiring = FullyConnected(units)
            
        # Replace LTCCell with LTCCustom
        self.rnn_cell = LTCCellCustom(wiring=wiring, in_features=input_size, input_mapping=input_mapping, output_mapping=output_mapping, ode_solver=ode_solver, 
                                      rtol=rtol, atol=atol, epsilon=epsilon, implicit_param_constraints=implicit_param_constraints)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   


# Inherit from your existing LTCCell so that we reuse the parameter allocation and mapping functions.
class GeneralizedLTCCell(LTCCell):
    def __init__(self, wiring:Wiring, solver:str='dopri5', **kwargs):
        super(GeneralizedLTCCell, self).__init__(wiring, **kwargs)
        self.solver = solver  # Store solver type (e.g., 'dopri5', 'rk4')

    def _ode_solver(self, inputs, state, elapsed_time):
        """
        Integrates the state using an adaptive ODE solver instead of Euler unfolding.
        This replaces the original _ode_solver in LTCCell.
        """
        v_pre = state
        
        # Compute sensory input (remains constant during ODE integration)
        sensory_w_activation = self.make_positive_fn(self._params["sensory_w"]) * self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"]) * self._params["sensory_sparsity_mask"]
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]
        
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        w_param = self.make_positive_fn(self._params["w"])
        gleak = self.make_positive_fn(self._params["gleak"])
        cm = self.make_positive_fn(self._params["cm"])

        def dynamics(t, v):
            """
            Defines the ODE system: d(v)/dt = f(v)
            The steady-state solution should match the original LTC's update rule.
            """
            cm_t = cm/(t + self._epsilon)
            w_activation = w_param * self._sigmoid(v, self._params["mu"], self._params["sigma"])*self._params["sparsity_mask"]

            rev_activation = w_activation * self._params["erev"]
            
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory


            numerator = cm_t * v + gleak * self._params["vleak"] + w_numerator
            denominator = cm_t + gleak + w_denominator

            v_target = numerator / (denominator + self._epsilon)

            return (v_target - v)

        if isinstance(elapsed_time, float):
            t_span = torch.tensor([0, elapsed_time], device=v_pre.device)
        else:
            t_span = elapsed_time
            t_span, indices = torch.sort(t_span)  # Ensure monotonic time
        
        v_out = odeint(dynamics, v_pre, t_span, method=self.solver)
        new_state = v_out[torch.argsort(indices)][-1] # Take the last state as the update result
        return new_state
    
    def forward(self, inputs, states, elapsed_time=1.0):
        # Regularly sampled mode (elapsed time = 1 second)
        inputs = self._map_inputs(inputs)

        next_state = self._ode_solver(inputs, states, elapsed_time)

        outputs = self._map_outputs(next_state)

        return outputs, next_state

# Now define the new GeneralizedLTC model that uses the GeneralizedLTCCell.
class GeneralizedLTC(LTC):
    def __init__(self, input_size:int, wiring:Wiring, solver:str='dopri5', return_sequences:bool=True, batch_first:bool=True, mixed_memory:bool=False, input_mapping:str="affine", output_mapping:str="affine", 
                 ode_unfolds:int=6, epsilon:float=1e-8, implicit_param_constraints:bool=True):
        """
        Generalized LTC Model using adaptive ODE integration.
        Inherits from LTC and replaces the cell with GeneralizedLTCCell.
        """
        super(GeneralizedLTC, self).__init__(input_size, wiring, return_sequences=return_sequences, batch_first=batch_first, mixed_memory=mixed_memory, input_mapping=input_mapping, output_mapping=output_mapping, 
                                             ode_unfolds=ode_unfolds, epsilon=epsilon, implicit_param_constraints=implicit_param_constraints)
        
        self.rnn_cell = GeneralizedLTCCell(self._wiring, solver=solver, in_features=input_size, input_mapping=input_mapping, output_mapping=output_mapping, 
                                           ode_unfolds=ode_unfolds, epsilon=epsilon, implicit_param_constraints=implicit_param_constraints)  # Use updated cell
        
    def forward(self, input, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0
        seq_dim = 1
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((seq_len, self.state_size), device=device)
            c_state = (torch.zeros((seq_len, self.state_size), device=device) if self.use_mixed else None)
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError("Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)")
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = (f"For batched 2-D input, hx and cx should also be 2-D but got ({h_state.dim()}-D) tensor")
                    raise RuntimeError(msg)
            else:
                # batchless  mode
                if h_state.dim() != 1:
                    msg = (f"For unbatched 1-D input, hx and cx should also be 1-D but got ({h_state.dim()}-D) tensor")
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for i in range(batch_size):
            inputs = input[i]
            ts = 1.0 if timespans is None else timespans[i].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(h_out)

        if self.return_sequences:
            stack_dim = 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        hx = (h_state, c_state) if self.use_mixed else h_state
        return readout, hx
    

class GeneralizedLiquidNeuralNetwork(Module):
    def __init__(self, n_neurons:int, neuron_type:GeneralizedLTC, in_features:int, out_features:int, hidden_size:int, store_hidden:bool=True, solver:str='rk4', seed:int=42) -> None:
        super(GeneralizedLiquidNeuralNetwork, self).__init__()
        
        self.ncp_wiring_output:AutoNCP = AutoNCP(n_neurons, hidden_size, seed=seed) # LNN
        self.ncp_layer_output:GeneralizedLTC = neuron_type(in_features-1, self.ncp_wiring_output, batch_first=True, solver=solver)
        self.out_layer = Linear(hidden_size, out_features)
        
        self.store_hidden = store_hidden
        self._reset_hidden_states()

    def _reset_hidden_states(self) -> 'GeneralizedLiquidNeuralNetwork':
        self.hidden = []
        return self
        
    def forward(self, x) -> Tensor:
        batch_size, window_size, n_features = x.shape
        t_span = x[:, :, 0]  # Extract time column
        x = x[:, :, 1:]  # Remove time column from input
        # x = x.view(-1, n_features - 1)  # Flatten batch and window size
        
        x, hidden = self.ncp_layer_output.forward(x, timespans=t_span)
        if self.store_hidden:
            self.hidden.append(hidden.cpu().detach().numpy())
        return self.out_layer.forward(x[:, -1, :])
    