from torch import Tensor
from typing import Callable

def euler(func:Callable[[Tensor, Tensor], Tensor], t:Tensor, dt:Tensor, y:Tensor) -> Tensor:
    return dt*func(t, y)

def rk4(func:Callable[[Tensor, Tensor], Tensor], t:Tensor, dt:Tensor, y:Tensor) -> Tensor:
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt, y + dt*k3)
    return (k1 + 2*(k2 + k3) + k4)*dt/6