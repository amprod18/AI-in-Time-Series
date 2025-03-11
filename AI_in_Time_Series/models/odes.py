import warnings
import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, Parameter, Tanh, ReLU
import warnings

from typing import *
from torch import Tensor


class ODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """General ODE representation for type annotations.

        Args:
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(ODEFunc, self).__init__()
        ...
        
    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        ...


class LinearODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """Optimized for systems whose evolution evolution is linear and are time independent.
        
        .. math:: 
            \\frac {dx} {dt} = A·x

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(LinearODEFunc, self).__init__()
        self.linear = Linear(in_features, out_features, bias=False)

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        return self.linear(x)


class NonLinearODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """Optimized for systems whose evolution is non-linear and are time independent.
        
        .. math:: 
            \\frac {dx} {dt} = f(A·x)
            f(x) = \\tanh(x)

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(NonLinearODEFunc, self).__init__()
        self.linear = Linear(in_features, out_features, bias=False)
        self.activation = Tanh()

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        return self.activation(self.linear(x))
    
    
class TimeDependentODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """Optimized for systems whose evolution is linear and are time dependent.
        
        .. math:: 
            \\frac {dx} {dt} = A·x(t)

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(TimeDependentODEFunc, self).__init__()
        self.linear = Linear(in_features+1, out_features)  # Include time

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        t_expanded = torch.full_like(x[:, :1], t.item())  # Repeat t for batch
        x_t = torch.cat([x, t_expanded], dim=1)  # Concatenate time
        return self.linear(x_t)
    
    
class DampedODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """Optimized for systems whose evolution time dependency is linear, time independent but unstable.
        
        .. math:: 
            \\frac {dx} {dt} = A·x·\\exp{-0.1·x}

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(DampedODEFunc, self).__init__()
        self.linear = Linear(in_features, out_features)

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        return -0.1 * x + self.linear(x)  # Damping term (-0.1 * x)
    
    
class ChaoticODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """Optimized for systems whose evolution is highly non-linear or chaotic and are time independent.
        
        .. math:: 
            \\frac {dx} {dt} = f(A·x)

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(ChaoticODEFunc, self).__init__()
        self.net = Sequential(
            Linear(in_features, hidden_size),
            ReLU(),
            Linear(hidden_size, out_features),
            Tanh()
        )

    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        return self.net(x)


class GeneralODEFunc(Module):
    def __init__(self, in_features:int, out_features:int, hidden_size:int) -> None:
        """General representation of an ODE which does not make any assumtions on the system evolution.
        
        .. math:: 
            \\frac {dx} {dt} = A·x + B·f(C·x)

        Parameters
        ----------
            hidden_size (int): Number of neurons of the linear representation of the ODE.
        """
        super(GeneralODEFunc, self).__init__()
        self.linear_term = Linear(in_features+1, out_features)  # Linear term
        self.nonlinear_term = Sequential(                    # Nonlinear term
            Linear(in_features+1, out_features),
            Tanh()
        )
        self.gate = Parameter(torch.ones(out_features))  # Learnable weight for nonlinearity
    
    def forward(self, t:Tensor, x:Tensor) -> Tensor:
        t_expanded = torch.full_like(x[:, :1], t.item())  # Repeat t for batch
        x_t = torch.cat([x, t_expanded], dim=1)  # Concatenate time
        linear_term = self.linear_term(x_t)
        nonlinear_term = self.nonlinear_term(x_t)
        return linear_term + self.gate * nonlinear_term


class SampleODEs:
    def __init__(self, **params) -> None:
        
        all_params = ['rate', 'mass', 'damping', 'stiffness', 'r', 'K', 'alpha', 'beta', 'delta', 'gamma', 'omega', 'mu']
        for param in all_params:
            setattr(self, param, None)
            
        for param, value in params.items():
            setattr(self, param, value)
    
    def exponential_decay(self, t, x):
        """ Exponential decay ODE: dx/dt = -rate * x """
        if self.rate is None:
            warnings.warn("rate not provided, using default value: 1.0")
            self.rate = 1.0
        return -self.rate * x

    def damped_oscillator(self, t, x):
        """ Damped harmonic oscillator: d^2x/dt^2 + (damping/mass) dx/dt + (stiffness/mass) x = 0 """
        if self.mass is None:
            warnings.warn("mass not provided, using default value: 1.0")
            self.mass = 1.0
        if self.damping is None:
            warnings.warn("damping not provided, using default value: 0.2")
            self.damping = 0.2
        if self.stiffness is None:
            warnings.warn("stiffness not provided, using default value: 1.0")
            self.stiffness = 1.0
        return torch.tensor([x[1], - (self.damping/self.mass) * x[1] - (self.stiffness/self.mass) * x[0]])

    def logistic_growth(self, t, x):
        """ Logistic growth: dx/dt = r * x * (1 - x/K) """
        if self.r is None:
            warnings.warn("growth rate r not provided, using default value: 1.0")
            self.r = 1.0
        if self.K is None:
            warnings.warn("carrying capacity K not provided, using default value: 10.0")
            self.K = 10.0
        return self.r * self.x * (1 - self.x / self.K)

    def predator_prey(self, t, x):
        """ Lotka-Volterra predator-prey equations """
        if self.alpha is None:
            warnings.warn("alpha not provided, using default value: 0.1")
            self.alpha = 0.1
        if self.beta is None:
            warnings.warn("beta not provided, using default value: 0.02")
            self.beta = 0.02
        if self.delta is None:
            warnings.warn("delta not provided, using default value: 0.01")
            self.delta = 0.01
        if self.gamma is None:
            warnings.warn("gamma not provided, using default value: 0.1")
            self.gamma = 0.1
        return torch.tensor([
            self.alpha * x[0] - self.beta * x[0] * x[1],
            self.delta * x[0] * x[1] - self.gamma * x[1]])

    def sine_wave(self, t, x):
        """ Simple sine wave oscillator: dx/dt = omega * cos(omega * t) """
        if self.omega is None:
            warnings.warn("omega not provided, using default value: 1.0")
            self.omega = 1.0
        return torch.tensor([self.omega * torch.cos(self.omega * t)])

    def van_der_pol(self, t, x):
        """ Van der Pol oscillator: d^2x/dt^2 - mu(1 - x^2)dx/dt + x = 0 """
        if self.mu is None:
            warnings.warn("mu not provided, using default value: 1.0")
            self.mu = 1.0
        return torch.tensor([x[1], self.mu * (1 - x[0]**2) * x[1] - x[0]])
