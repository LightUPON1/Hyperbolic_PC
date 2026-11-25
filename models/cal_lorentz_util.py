import torch
from typing import Tuple, Any, Union, List

max_norm = 85
eps = 1e-15

def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-16)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)

class GradClippedACos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        y = torch.acos(input)
        ctx.save_for_backward(input)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = 1e-6
        x_clip, = ctx.saved_tensors
        in_sqrt = 1. - torch.pow(x_clip, 2)
        in_sqrt += eps
        denom = torch.sqrt(in_sqrt)
        denom += eps
        dydx = -1. / denom
        return grad_output * dydx

def acos(x):
    # https://github.com/jdieter31/riemannian-nlp/blob/75ef47608c81ec6e925fe24d16c67985e4b987c6/riemann/manifolds/sphere.py#L10
    # https: // github.com / ziyc / L - Tracing / blob / c82fcb3df42bf1733223e86443e07bcad6046d5a / utils / math.py  # L75
    return GradClippedACos.apply(x)

class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        with torch.no_grad():
            ctx.save_for_backward(x.ge(min) & x.le(max))
            return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None

def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)
class Acosh(torch.autograd.Function):
    """
    Numerically stable arccosh that never returns NaNs.
    Returns acosh(x) = arccosh(x) = log(x + sqrt(max(x^2 - 1, eps))).
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = clamp(x, min=1 + eps)
            z = sqrt(x * x - 1.)
            ctx.save_for_backward(z)
            return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        # z_ = clamp(z, min=eps)
        z_ = z
        return grad_output / z_


def acosh(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable arccosh that never returns NaNs.

    :param x: The input tensor.
    :return: log(x + sqrt(max(x^2 - 1, eps))
    """
    return Acosh.apply(x)
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
def artanh(x):
    return Artanh.apply(x)

