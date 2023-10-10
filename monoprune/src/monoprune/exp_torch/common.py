from monoprune.env import *

from torch import Tensor
import torch
import torch.nn.functional as F
from jaxtyping import Float, Bool, Int, jaxtyped
from monoprune.interval import Interval
from monoprune.exp_synth_param.common import (
    heuristic_search_key as heuristic_search_key,
    breadth_first_search_key as breadth_first_search_key,
    online_search as online_search,
    ExperimentOutputLine as ExperimentOutputLine,
)
import sys


def f1_pre(
    trues: Bool[Tensor, "#*b n"], preds: Bool[Tensor, "#*b n"]
) -> Tuple[Int[Tensor, "#*b"], Int[Tensor, "#*b"], Int[Tensor, "#*b"],]:
    tp = preds[trues].sum()
    fp = preds[~trues].sum()
    np = trues.sum()
    return tp, fp, np  # type: ignore


def f1(
    trues: Bool[Tensor, "#*b n"], preds: Bool[Tensor, "#*b n"]
) -> Float[Tensor, "#*b"]:
    tp, fp, np = f1_pre(trues, preds)
    f1_ = 2 * tp / (tp + fp + np)
    assert (f1_ == f1_).all()
    return f1_


def f1_interval(
    trues: Bool[Tensor, "#*b n"],
    preds_lower: Bool[Tensor, "#*b n"],
    preds_upper: Bool[Tensor, "#*b n"],
) -> Tuple[Float[Tensor, "#*b"], Float[Tensor, "#*b"]]:
    assert (preds_lower <= preds_upper).all()
    tp_lower, fp_lower, np = f1_pre(trues, preds_lower)
    tp_upper, fp_upper, _ = f1_pre(trues, preds_upper)
    f1_lower = 2 * tp_lower / (tp_lower + fp_upper + np)
    assert (f1_lower == f1_lower).all()
    f1_upper = 2 * tp_upper / (tp_upper + fp_lower + np)
    assert (f1_upper == f1_upper).all()
    return f1_lower, f1_upper


DO_DBG = True


def bce_logit(logit, target, weight=None, reduction="mean"):
    assert target.dtype == torch.bool
    return F.binary_cross_entropy_with_logits(
        logit, target.float(), weight, reduction=reduction
    )


def bce_logit_interval(logit_lower, logit_upper, target, weight=None, reduction="mean"):
    assert target.dtype == torch.bool

    # bce_logit runs batched, and so each input is independent
    # if the label for that input is true, then it's antitone
    # (since it's a loss) else its monotone
    logit_mod_lower = torch.where(target, logit_upper, logit_lower)
    logit_mod_upper = torch.where(target, logit_lower, logit_upper)

    ret_lower = bce_logit(logit_mod_lower, target, weight, reduction)
    ret_upper = bce_logit(logit_mod_upper, target, weight, reduction)

    if DO_DBG:
        check = bce_logit((logit_lower + logit_upper) / 2, target, weight, reduction)
        assert ret_lower.shape == check.shape
        assert (ret_lower <= check).all()
        assert (ret_upper >= check).all()
    return ret_lower, ret_upper


if False:
    X_lower = torch.tensor([0.0, -10.0])
    X_upper = torch.tensor([10.0, -10.0])
    y = torch.tensor([True, False])
    bce_logit_interval(X_lower, X_upper, y)

if False:

    def binary_crossent(
        logits: Float[Tensor, "#*b n"],
        truth: Bool[Tensor, "#*b n"],
        weight: Float[Tensor, "2"],
    ) -> Float[Tensor, "#*b n"]:
        logits_aug = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
        logits_norm = F.log_softmax(logits_aug, -1)
        logits_weighted = weight * logits_norm
        ret = -torch.where(truth, logits_weighted[:, 0], logits_weighted[:, 1])
        if DO_DBG:
            check = F.cross_entropy(
                logits_aug, truth.long(), weight=weight[..., [1, 0]], reduction="none"
            )
            assert check.shape == ret.shape
            diff = torch.max(torch.abs(check - ret))
            assert diff == 0.0, (diff, check, ret)
        return ret

    def binary_crossent_interval(
        logits_lower: Float[Tensor, "#*b n"],
        logits_upper: Float[Tensor, "#*b n"],
        truth: Bool[Tensor, "#*b n"],
        weight: Float[Tensor, "2"],
    ) -> Tuple[Float[Tensor, "#*b n"], Float[Tensor, "#*b n"]]:
        logits_aug_lower = torch.stack(
            [logits_lower, torch.zeros_like(logits_lower)], dim=-1
        )
        logits_norm_lower = F.log_softmax(logits_aug_lower, -1)
        logits_weighted_lower = weight * logits_norm_lower
        logits_aug_upper = torch.stack(
            [logits_upper, torch.zeros_like(logits_upper)], dim=-1
        )
        logits_norm_upper = F.log_softmax(logits_aug_upper, -1)
        logits_weighted_upper = weight * logits_norm_upper
        unreduced_lower = -torch.where(
            truth, logits_weighted_upper[:, 0], logits_weighted_lower[:, 1]
        )
        unreduced_upper = -torch.where(
            truth, logits_weighted_lower[:, 0], logits_weighted_upper[:, 1]
        )
        if DO_DBG:
            assert (logits_lower <= logits_upper).all()
            logits_mid = (logits_lower + logits_upper) / 2
            logits_aug_mid = torch.stack(
                [logits_mid, torch.zeros_like(logits_mid)], dim=-1
            )
            logits_norm_mid = F.log_softmax(logits_aug_mid, -1)
            ret_mid = -torch.where(truth, logits_norm_mid[:, 0], logits_norm_mid[:, 1])
            # ret_mid = binary_crossent((logits_lower + logits_upper) / 2, truth)
            # print("aug")
            # print(logits_aug_lower)
            # print(logits_aug_mid)
            # print(logits_aug_upper)
            # print("norm")
            # print(logits_norm_lower)
            # print(logits_norm_mid)
            # print(logits_norm_upper)
            # print("unreduced")
            # print(unreduced_lower)
            # print(ret_mid)
            # print(unreduced_upper)
            lower_succs = unreduced_lower <= ret_mid
            upper_succs = unreduced_upper >= ret_mid
            assert lower_succs.all(), (
                (~lower_succs).sum(),
                unreduced_lower[~lower_succs],
                ret_mid[~lower_succs],
                logits_lower[~lower_succs],
                logits_upper[~lower_succs],
            )
            assert upper_succs.all()
        return unreduced_lower, unreduced_upper

    def maybe_bce(logit, target, weight=None):
        logits_aug = torch.stack([torch.zeros_like(logit), logit], dim=-1)
        return F.cross_entropy(logits_aug, target.long(), weight)

    X = torch.tensor([10.0, -10.0])
    y = torch.tensor([True, False])
    F.binary_cross_entropy_with_logits(X, y.float())
    maybe_bce(X, y)

    if False:
        logit_lower = torch.tensor([-435.7541])
        logit_upper = torch.tensor([0.9956])
        truth = torch.tensor([False])
        weight = torch.tensor([1.0, 1.0])
        logit_mid = (logit_lower + logit_upper) / 2
        logit_aug = torch.stack([logit_mid, torch.zeros_like(logit_mid)], dim=-1)
        logit_aug
        F.cross_entropy(logit_aug, truth.long(), weight=weight, reduction="none")
        binary_crossent(logit_mid, truth, weight)
        binary_crossent_interval(
            logit_lower,
            logit_upper,
            truth,
            weight=weight,
        )

import itertools


def make_all_boolean_combinations(d: int) -> Iterable[Tuple[bool]]:
    for bs in itertools.product((False, True), repeat=d):
        yield bs


@jaxtyped
def split_interval_context(
    lb: Float[Tensor, "p"],
    ub: Float[Tensor, "p"],
) -> Iterable[Tuple[Float[Tensor, "p"], Float[Tensor, "p"]]]:
    for kb in make_all_boolean_combinations(lb.shape[-1]):
        mask = torch.tensor(kb, dtype=torch.bool)
        midpoint: Float[Tensor, "p"] = (lb + ub) / 2
        new_lb = torch.where(mask, midpoint, lb)
        new_ub = torch.where(mask, ub, midpoint)
        yield (new_lb, new_ub)


IntervalState: TypeAlias = Tuple[T, Float[Tensor, "p"], Float[Tensor, "p"]]
ConcreteUtil: TypeAlias = Callable[[T, Float[Tensor, "p"]], float]
IntervalUtil: TypeAlias = Callable[
    [T, Float[Tensor, "p"], Float[Tensor, "p"]], Tuple[float, float]
]


def state_evaluator(
    concrete_util: ConcreteUtil[T],
    interval_util: Optional[IntervalUtil[T]],
) -> Callable[[IntervalState[T]], Interval[rat]]:
    def _eval_state(state: IntervalState[T]) -> Interval[rat]:
        midpoint: Float[Tensor, "p"] = (state[1] + state[2]) / 2
        midpoint_output = concrete_util(state[0], midpoint)

        if interval_util is not None:
            interval_output_lower, interval_output_upper = interval_util(
                state[0], state[1], state[2]
            )
            assert midpoint_output >= interval_output_lower
            assert midpoint_output <= interval_output_upper
        else:
            # TODO: take initial bounds as input
            interval_output_upper = 1.0
            # interval_output_upper = sys.float_info.max
        return Interval(
            rat.from_float(midpoint_output), rat.from_float(interval_output_upper)
        )

    return _eval_state


def expand_state(state: IntervalState[T]) -> Iterable[IntervalState[T]]:
    for new_lb, new_ub in split_interval_context(state[1], state[2]):
        yield (state[0], new_lb, new_ub)


""" 
def heuristic_opt(
    sketches_with_bounds: Iterable[IntervalState[T]],
) -> Iterable[ExperimentOutputLine[IntervalState[T]]]:
    return online_search(
        heuristic_search_key(),
        _state_evaluator(True),
        _expand_state,
        sketches_with_bounds,
    )


def breadth_first_opt(
    sketches_with_bounds: Iterable[IntervalState[T]],
) -> Iterable[ExperimentOutputLine[IntervalState[T]]]:
    return online_search(
        breadth_first_search_key(),
        _state_evaluator(True),
        _expand_state,
        sketches_with_bounds,
    )


def dyadic2_opt(
    sketches_with_bounds: Iterable[IntervalState[T]],
) -> Iterable[common.ExperimentOutputLine[IntervalState[T]]]:
    return common.online_search(
        common.breadth_first_search_key(),
        _state_evaluator(False),
        _expand_state,
        sketches_with_bounds,
    )
 """


def make_subset_indices(seed: int, n_tot: int, n_sub: int):
    torch.manual_seed(seed)
    perm = torch.randperm(n_tot)
    return perm[0:n_sub]
