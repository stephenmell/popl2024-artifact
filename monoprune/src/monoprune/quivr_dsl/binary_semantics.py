from monoprune.env import *
from monoprune.quivr_dsl.syntax import *
from monoprune.maxmin_utils import disjconj_matvecmul

import torch
from torch import Tensor
from jaxtyping import (
    Float,
    Bool,
    Int,
    jaxtyped,
)

Trace: TypeAlias = Tuple[
    Dict[str, Bool[Tensor, "n n"]],
    Dict[str, Float[Tensor, "n n"]],
]

BatchedTrace: TypeAlias = Tuple[
    Dict[str, Bool[Tensor, "bt n n"]],
    Dict[str, Float[Tensor, "bt n n"]],
]


def batched_trace_shape(x: BatchedTrace) -> Tuple[int, ...]:
    pred_0, _pred_1 = x
    assert len(pred_0) != 0
    return next(iter(pred_0.values())).shape


def batched_trace_device(x: BatchedTrace) -> torch.device:
    pred_0, _pred_1 = x
    assert len(pred_0) != 0
    return next(iter(pred_0.values())).device


def fold_taking_first(op: Callable[[T, T], T], iter: Iterable[T]) -> T:
    ret = None
    have_first = False
    for x in iter:
        if not have_first:
            have_first = True
            ret = x
        else:
            ret = op(ret, x)  # type: ignore
    assert have_first
    return ret  # type: ignore


@jaxtyped
def quadratic_eval_bool(
    traces: BatchedTrace,
    threshold: Float[Tensor, "p"],
    initial_expr: QuivrExpr[Tuple[int, ...]],
    initial_rvec: Bool[Tensor, "bt n"],
) -> Bool[Tensor, "bt n"]:
    def rec(
        expr: QuivrExpr[Tuple[int, ...]],
        rvec: Bool[Tensor, "bt n"],
    ) -> Bool[Tensor, "bt n"]:
        match expr:
            case "pred", (name, data):
                match data:
                    case ():
                        pred_mat = traces[0][name]
                    case (param_index,):
                        pred_mat_raw = traces[1][name] >= threshold[param_index]
                        pred_mat = torch.triu(pred_mat_raw)
                        # assert (pred_mat == pred_mat_raw).all(), (name, data)
                    case x:
                        assert False, x
                ret = disjconj_matvecmul(
                    pred_mat,
                    rvec,
                )

            case ("seq", subexprs):
                cur_vec = rvec
                subexpr: QuivrExpr[Tuple[int, ...]]
                for subexpr in reversed(subexprs):  # type: ignore
                    cur_vec = rec(subexpr, cur_vec)
                ret = cur_vec

            case ("conj", subexprs):
                ret = fold_taking_first(
                    lambda a, b: a & b, (rec(subexpr, rvec) for subexpr in subexprs)  # type: ignore
                )
        return ret

    return rec(initial_expr, initial_rvec)


@jaxtyped
def eval_bool(
    traces: BatchedTrace,
    threshold: Float[Tensor, "p"],
    initial_expr: QuivrExpr[Tuple[int, ...]],
) -> Bool[Tensor, "bt"]:
    rvec = torch.empty(
        batched_trace_shape(traces)[:-1],
        dtype=torch.bool,
        device=batched_trace_device(traces),
    )
    rvec[..., :-1] = False
    rvec[..., -1] = True

    ret = quadratic_eval_bool(traces, threshold, initial_expr, rvec)[..., 0]

    return ret
