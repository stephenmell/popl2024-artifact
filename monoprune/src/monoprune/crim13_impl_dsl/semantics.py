from monoprune.env import *
from monoprune.torch_intervals import (
    batched_affine, batched_affine_interval, batched_ite_interval, mul_interval
)
from monoprune.crim13_impl_dsl.syntax import (
    CrimExprAtomAtom,
    CrimExprAtomAtomIndexed,
    CrimFeatureName,
)

import torch
from torch import Tensor
from jaxtyping import (
    Float,
    Bool,
    Int,
    jaxtyped,
)

feature_index_table: Dict[str, Tuple[int, ...]] = {
    "position": (0, 1, 2, 3),
    "distance": (4,),
    "distance_change": (5,),
    "velocity": (
        11,
        12,
        13,
        14,
    ),
    "acceleration": (
        15,
        16,
        17,
        18,
    ),
    "angle": (6, 7, 10),
    "angle_change": (8, 9),
}
@jaxtyped
def eval_atom_atom(
    traces: Float[Tensor, "#*b trace time in"],
    all_parameters: Float[Tensor, "param"],
    # extra_context_size: int,
    output_size: int,
    initial_expr: CrimExprAtomAtomIndexed,
) -> Float[Tensor, "#*b trace time out"]:
    def rec(
        expr: CrimExprAtomAtomIndexed,
    ) -> Float[Tensor, "#*b trace time out"]:
        match expr:
            case "ite", (e_c, e_a, e_b):
                a = rec(e_a)
                b = rec(e_b)
                c = rec(e_c)
                ret = torch.where(c >= 0, a, b)
            case "add", (e_a, e_b):
                a = rec(e_a)
                b = rec(e_b)
                ret = a + b
            case "mul", (e_a, e_b):
                a = rec(e_a)
                b = rec(e_b)
                ret = a * b
            case "sel", (feat_name, data):
                assert output_size == 1
                parameters = all_parameters[data,]
                weights = parameters[1:]
                bias = parameters[0]
                feat_vals = traces[..., feature_index_table[feat_name]]
                ret = batched_affine(weights, feat_vals, bias)

        return ret

    return rec(initial_expr)

DO_DBG = True
@jaxtyped
def eval_atom_atom_interval(
    traces: Float[Tensor, "#*b trace time in"],
    all_parameters_lower: Float[Tensor, "param"],
    all_parameters_upper: Float[Tensor, "param"],
    # extra_context_size: int,
    output_size: int,
    initial_expr: CrimExprAtomAtomIndexed,
) -> Tuple[Float[Tensor, "#*b trace time out"], Float[Tensor, "#*b trace time out"]]:
    def rec(
        expr: CrimExprAtomAtomIndexed,
    ) -> Tuple[
        Float[Tensor, "#*b trace time out"], Float[Tensor, "#*b trace time out"]
    ]:
        match expr:
            case "ite", (e_c, e_a, e_b):
                a_lower, a_upper = rec(e_a)
                b_lower, b_upper = rec(e_b)
                c_lower, c_upper = rec(e_c)
                ret = batched_ite_interval(c_lower, c_upper, a_lower, a_upper, b_lower, b_upper)
            case "add", (e_a, e_b):
                a_lower, a_upper = rec(e_a)
                b_lower, b_upper = rec(e_b)
                ret = (a_lower + b_lower, a_upper + b_upper)
            case "mul", (e_a, e_b):
                a_lower, a_upper = rec(e_a)
                b_lower, b_upper = rec(e_b)
                ret = mul_interval(a_lower, a_upper, b_lower, b_upper)
            case "sel", (feat_name, data):
                assert output_size == 1
                parameters_lower = all_parameters_lower[data,]
                parameters_upper = all_parameters_upper[data,]
                weights_lower = parameters_lower[1:]
                weights_upper = parameters_upper[1:]
                bias_lower = parameters_lower[0]
                bias_upper = parameters_upper[0]
                feat_vals = traces[..., feature_index_table[feat_name]]
                ret = batched_affine_interval(
                    weights_lower, weights_upper, feat_vals, bias_lower, bias_upper
                )
        if DO_DBG:
            ret_midpoint = eval_atom_atom(traces, (all_parameters_lower + all_parameters_upper)/2, output_size, expr)
            assert (ret_midpoint >= ret[0]).all()
            assert (ret_midpoint <= ret[1]).all()

        return ret

    return rec(initial_expr)
