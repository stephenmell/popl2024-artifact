from typing import TypeAlias
from monoprune.env import *
from monoprune.aexpr.symbolic_syntax import (
    BoolSymbolic,
    IntSymbolic,
    RatSymbolic,
    symbolic_int_lit,
    symbolic_rat_lit,
    symbolic_rat_var,
)


def f1_pre(
    trues: Tuple[bool, ...], preds: Tuple[BoolSymbolic, ...]
) -> Tuple[IntSymbolic, IntSymbolic, IntSymbolic]:
    tp = symbolic_int_lit(0)
    fp = symbolic_int_lit(0)
    np = symbolic_int_lit(0)
    for true, pred in zip(trues, preds):
        if true:
            np += 1
            tp += ite(pred, 1, 0)
        else:
            fp += ite(pred, 1, 0)

    return tp, fp, np


def f1_div(trues: Tuple[bool, ...], preds: Tuple[BoolSymbolic, ...]) -> RatSymbolic:
    tp, fp, np = f1_pre(trues, preds)

    ret = (2 * tp) / (tp + fp + np)

    return ret


def f1_sb(trues: Tuple[bool, ...], preds: Tuple[BoolSymbolic, ...]) -> RatSymbolic:
    tp, fp, np = f1_pre(trues, preds)

    s0 = tp / (fp + np)
    s1 = RatSymbolic(("sb_right", (s0.t,)))
    ret = 2 * s1

    return ret

