from monoprune.env import *
from monoprune.torch_intervals import (
    batched_affine, batched_affine_interval, batched_ite_interval, mul_interval
)
from monoprune.crim13_paper_dsl.syntax import (
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


@jaxtyped
def eval_atom_atom(
    traces: Float[Tensor, "#*b trace time in"],
    all_parameters: Float[Tensor, "param"],
    initial_expr: CrimExprAtomAtomIndexed,
) -> Float[Tensor, "#*b trace time"]:
    def rec(
        expr: CrimExprAtomAtomIndexed,
    ) -> Float[Tensor, "#*b trace time"]:
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
            case "sel", (feat_name,):
                ret = traces[..., feat_name]
            case "lit", (data):
                ret = all_parameters[data]

        return ret

    return rec(initial_expr)


DO_DBG = True


@jaxtyped
def eval_atom_atom_interval(
    traces: Float[Tensor, "#*b trace time in"],
    all_parameters_lower: Float[Tensor, "param"],
    all_parameters_upper: Float[Tensor, "param"],
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
                ret = batched_ite_interval(
                    c_lower, c_upper, a_lower, a_upper, b_lower, b_upper
                )
            case "add", (e_a, e_b):
                a_lower, a_upper = rec(e_a)
                b_lower, b_upper = rec(e_b)
                ret = (a_lower + b_lower, a_upper + b_upper)
            case "mul", (e_a, e_b):
                a_lower, a_upper = rec(e_a)
                b_lower, b_upper = rec(e_b)
                ret = mul_interval(a_lower, a_upper, b_lower, b_upper)
            case "sel", (feat_name,):
                feat = traces[..., feat_name]
                ret = (feat, feat)
            case "lit", (data):
                ret = (all_parameters_lower[data], all_parameters_upper[data])
        if DO_DBG:
            ret_midpoint = eval_atom_atom(
                traces,
                (all_parameters_lower + all_parameters_upper) / 2,
                expr,
            )
            assert (ret_midpoint >= ret[0]).all()
            assert (ret_midpoint <= ret[1]).all()

        return ret

    return rec(initial_expr)


if False:
    from monoprune.crim13_paper_dsl.syntax import atom_atom_add_param_indices

    outline0: CrimExprAtomAtom[None] = (
        "add",
        (
            (
                "mul",
                (
                    ("sel", (0,)),
                    ("lit", (None,)),
                ),
            ),
            ("lit", (None,)),
        ),
    )
    expr0, paramlist0 = atom_atom_add_param_indices(outline0, 0)
    assert paramlist0 == 2
    data = torch.tensor(
        [
            [[0.0]],
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[4.0]],
        ]
    )
    params = torch.tensor([1.0, -2.0])
    res = eval_atom_atom(data, params, expr0)
    print(res)
