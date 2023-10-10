from monoprune.env import *
import torch
from torch import Tensor
from jaxtyping import jaxtyped, Float


@jaxtyped
def batched_affine(
    weight: Float[Tensor, "#*b n"],
    x: Float[Tensor, "#*b n"],
    bias: Float[Tensor, "#*b"],
) -> Float[Tensor, "#*b"]:
    return torch.sum(weight * x, -1) + bias


@jaxtyped
def mul_interval(
    a_lower: Float[Tensor, "#*b"],
    a_upper: Float[Tensor, "#*b"],
    b_lower: Float[Tensor, "#*b"],
    b_upper: Float[Tensor, "#*b"],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    ll = a_lower * b_lower
    lr = a_lower * b_upper
    rl = a_upper * b_lower
    rr = a_upper * b_upper
    ret_lower = torch.minimum(torch.minimum(ll, lr), torch.minimum(rl, rr))
    ret_upper = torch.maximum(torch.maximum(ll, lr), torch.maximum(rl, rr))
    return ret_lower, ret_upper


@jaxtyped
def pow_interval(
    a_lower: Float[Tensor, "#*b"],
    a_upper: Float[Tensor, "#*b"],
    n: int,
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    if n % 2 == 0:
        # three cases: interval all positive, interval all negative, mixed
        # all pos: just pow the upper and lower bounds
        # all neg: pow bounds and flip them
        # mixed: lb is zero, ub is max of pos of ub, lb
        ret_lower = torch.where(
            a_lower >= 0.0,
            torch.pow(a_lower, n),
            torch.where(
                a_upper <= 0.0,
                torch.pow(a_upper, n),
                torch.zeros_like(a_lower),
            ),
        )
        ret_upper = torch.where(
            a_lower >= 0.0,
            torch.pow(a_upper, n),
            torch.where(
                a_lower <= 0.0,
                torch.pow(a_lower, n),
                torch.maximum(torch.pow(a_lower, n), torch.pow(a_upper, n)),
            ),
        )
    else:
        ret_lower = torch.pow(a_lower, n)
        ret_upper = torch.pow(a_upper, n)

    return ret_lower, ret_upper


@jaxtyped
def batched_affine_interval(
    weight_lower: Float[Tensor, "#*b n"],
    weight_upper: Float[Tensor, "#*b n"],
    x: Float[Tensor, "#*b n"],
    bias_lower: Float[Tensor, "#*b"],
    bias_upper: Float[Tensor, "#*b"],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    mul_lower, mul_upper = mul_interval(weight_lower, weight_upper, x, x)
    ret_lower = torch.sum(mul_lower, -1) + bias_lower
    ret_upper = torch.sum(mul_upper, -1) + bias_upper
    return ret_lower, ret_upper


@jaxtyped
def batched_ite_interval(
    c_lower: Float[Tensor, "#*b"],
    c_upper: Float[Tensor, "#*b"],
    a_lower: Float[Tensor, "#*b n"],
    a_upper: Float[Tensor, "#*b n"],
    b_lower: Float[Tensor, "#*b n"],
    b_upper: Float[Tensor, "#*b n"],
):
    c2_lower = (c_lower >= 0).unsqueeze(-1)
    c2_upper = (c_upper >= 0).unsqueeze(-1)

    case_t_lower = a_lower
    case_t_upper = a_upper
    case_f_lower = b_lower
    case_f_upper = b_upper
    case_ft_lower = torch.minimum(a_lower, b_lower)
    case_ft_upper = torch.maximum(a_upper, b_upper)
    ret_lower = torch.where(
        c2_lower,  # condition is always true
        case_t_lower,
        torch.where(
            ~c2_upper,  # condition is always true
            case_f_lower,
            case_ft_lower,
        ),
    )
    ret_upper = torch.where(
        c2_upper,  # condition is always true
        case_t_upper,
        torch.where(
            ~c2_upper,  # condition is always true
            case_f_upper,
            case_ft_upper,
        ),
    )
    return ret_lower, ret_upper
