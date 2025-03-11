import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchdiffeq import odeint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from .CustomDatatypes import DatasetStorer
from ..models.odes import SampleODEs

from IPython.core.display import HTML
from IPython.display import display

from typing import *
from pandera.typing import DataFrame, Series
from torch import Tensor
from numpy import ndarray


def _apply_scalers(X:DataFrame, y:Union[DataFrame,Series], scalers:Union[tuple[Union[MinMaxScaler,StandardScaler], Union[MinMaxScaler,StandardScaler]],None]=None) -> tuple[DataFrame, Union[DataFrame,Series], Union[MinMaxScaler,StandardScaler], Union[MinMaxScaler,StandardScaler]]:
    if len(X) == 0:
        raise ValueError("Please eprovide a valid dataframe with at least two points.")
    if scalers is None:
        sc_x = MinMaxScaler()
        sc_y = MinMaxScaler()
        X_scaled = pd.DataFrame(sc_x.fit_transform(X), columns=X.columns, index=X.index)
        y_scaled = pd.DataFrame(sc_y.fit_transform(y), columns=y.columns, index=y.index)
    else:
        sc_x = scalers[0]
        sc_y = scalers[1]
        X_scaled = pd.DataFrame(sc_x.transform(X), columns=X.columns, index=X.index)
        y_scaled = pd.DataFrame(sc_y.transform(y), columns=y.columns, index=y.index)
        
    return X_scaled, y_scaled, sc_x, sc_y
    
def create_synthetic_dataset(dataset_parameters:Dict[str, Any], display_heads:bool=False, plot_result:bool=False) -> DatasetStorer:
    """
    Parameters
    ----------
        dataset_parameters: Dict[str, Any] 
            Dictionary containing the hyperparameters that reference the dataset creation.
        - ode_func: Callable[[Tensor, Tensor], Tensor]
            - Ode representation in a callable form which take 't' and 'x' as inputs. Defaults to SampleODEs.exponential_decay.
        - start_t: float
            - Start time for the dataset. Defaults to 0.
        - max_t: float
            - End time for the dataset. Defaults to 10.
        - n_points: int
            - Number of points to integrate within the timespan provided. Defaults to 1000.
        - look_ahead: int
            - Number of steps to look forward for the target generation. Defaults to 1.
        - initial_guess: Union[ndarray[float,Any], Tensor, list[float]]
            - Initial values to start the integration (aka, f(t=start_t)). Defaults to [1, 1].
        - method: str
            - Integration method to use. Defaults to rk4.
        - ode_params: Dicts[str, Any]
            - Hyperparameters that define the ODE
        
        display_heads: bool, optional
            Whether to display the heads of the different partitions performed. Defaults to False.
        plot_result: bool, optional) 
            Whether to plot the resulting dataset in a pairplot form. Defaults to False.

    Returns
    -------
        Stored dataset: DatasetStorer
            Returns a custom class which contains all the hyperparameters used in the dataset creation (provided and computed) and also the datasets themselves.
    """
    dataset_points = generate_synthetic_dataset(**dataset_parameters)
    features = [f'Component{dimension}' for dimension in range(1, len(dataset_parameters['initial_guess'])+1)]
    targets = [f'Component{dimension}_target' for dimension in range(1, len(dataset_parameters['initial_guess'])+1)]
    
    dataset_df = pd.DataFrame(dataset_points, columns=['t'] + features)
    if plot_result:
        sns.pairplot(data=dataset_df)
        plt.show()
    
    for feature in features:
        dataset_df.loc[:, [f'{feature}_target']] = dataset_df[feature].shift(-dataset_parameters['look_ahead'])
    features = ['t'] + features
    
    X, y = dataset_df[features], dataset_df[targets]

    train_size = 0.8

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)
    X_train_scaled, y_train_scaled, sc_x, sc_y = _apply_scalers(X_train, y_train, scalers=None)
    X_test_scaled, y_test_scaled, sc_x, sc_y = _apply_scalers(X_test, y_test, scalers=(sc_x, sc_y))

    print(X_train_scaled.shape, y_train_scaled.shape, X_test_scaled.shape, y_test_scaled.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    if display_heads:
        for df in [X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_train, y_train, X_test, y_test]:
            display(HTML(df.tail().to_html()))
    
    hyperparameters = dict(zip(['features', 'targets', 'train_size'], [features, targets, train_size]))
    hyperparameters.update(dataset_parameters)
    
    dataset = DatasetStorer('main', **hyperparameters)
    dataset = dataset.add_container('data', **dict(zip(['X_train_scaled', 'y_train_scaled', 'X_test_scaled', 'y_test_scaled', 'X_train', 'y_train', 'X_test', 'y_test'], [X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_train, y_train, X_test, y_test])))
    dataset = dataset.add_container('scalers', **dict(zip(['sc_x', 'sc_y'], [sc_x, sc_y])))
    
    return dataset
    
def generate_synthetic_dataset(ode_func:Callable[[Tensor, Tensor], Tensor]=SampleODEs.exponential_decay, initial_guess:Union[ndarray[float,Any], Tensor, list[float]]=[1., 1.], start_t:float=0, max_t:float=10, n_points:int=1000, look_ahead:int=1, method='rk4', ode_params:Union[Dict[str,Any],None]=None) -> Tensor:
    """
    Generates a synthetic dataset by solving an ODE.
    
    Parameters
    ----------
        ode_func: Callable[[Tensor, Tensor], Tensor]
            The ODE function dx/dt = f(x, t).
        initial_guess: Union[ndarray[float,Any], Tensor, list[float]]: 
            Initial condition for the ODE.
        start_t: float 
            Start time for the ode integration.
        max_t: float 
            End time for the ode integration.
        n_points: int 
            Number of points to integrate in the time span provided.
        method: str 
            Numerical integration method ('euler', 'rk4', etc.).
        as_df bool: 
            Set to True to return the data as a pandas dataframe instead
        ode_params: Tensor
            Additional parameters of the ode to be integrated.
    
    Returns
    -------
        integrated points: Union[Tensor,DataFrame[float]]
            Solution of the ODE at discrete time points.
    """
    if len(initial_guess) == 0:
        raise ValueError("Provide a valid initial guess, referencing the number of dimensions of you ODE.")
    if not callable(ode_func):
        raise AttributeError("Please provide a callable function as the ODE to be integrated.")
    
    if ode_params is not None:
        ode = SampleODEs(**ode_params)
    else:
        ode = SampleODEs()
    
    try:
        ode_func_ = getattr(ode, ode_func.__name__)
    except AttributeError:
        ode_func_ = ode_func

    step_size = (max_t - start_t)/n_points
    t_vals = torch.arange(start_t, max_t, step_size)
    
    if method == 'euler':
        solutions = [initial_guess]
        x = initial_guess
        for t in t_vals[:-1]:
            x = x + step_size * ode_func_(t, x)
            solutions.append(x)
        solutions = torch.stack(solutions)
    else:
        # Use torchdiffeq.odeint for advanced solvers
        solutions = odeint(ode_func_, initial_guess, t_vals, method=method)
    
    return torch.cat([t_vals.reshape(-1, 1), solutions], axis=1)