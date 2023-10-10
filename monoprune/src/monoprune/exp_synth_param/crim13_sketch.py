from typing import TypeAlias
from monoprune.env import *
from monoprune.exp_synth_param.crim13_data import get_train
from monoprune.aexpr.symbolic_syntax import (
    BoolSymbolic,
    IntSymbolic,
    RatSymbolic,
    symbolic_int_lit,
    symbolic_rat_lit,
    symbolic_rat_var,
)
from monoprune.aexpr.semantics_concrete import semantics_concrete
from monoprune.aexpr.semantics_interval import semantics_interval
from monoprune.aexpr.semantics_interval_sampling import semantics_interval_sampling
from monoprune.aexpr.semantics_z3 import semantics_z3
from monoprune.unique import unique
from monoprune.aexpr.syntax import RatTerm
from monoprune.interval import Interval
from monoprune.exp_synth_param.smt import smt_opt
from monoprune.exp_synth_param.dyadic import dyadic_opt
from monoprune.exp_synth_param.common import ctime

Param: TypeAlias = Tuple[RatSymbolic]

import pickle
import sys


def eval_sketch(feature: int, param: Param, x: Tuple[rat, ...]) -> BoolSymbolic:
    return x[feature] <= param[0]


from monoprune.stat_metrics import f1_sb


def util(
    eval_sketch: Callable[[Any, Tuple[rat, ...]], BoolSymbolic],
    train_X: Tuple[Tuple[rat, ...], ...],
    train_y: Tuple[bool, ...],
    param: Param,
) -> RatSymbolic:
    preds = tuple(eval_sketch(param, x) for x in train_X)
    return f1_sb(train_y, preds)


import resource

resource.setrlimit(
    resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
)

param_c_name = unique("c")
param: Param = (symbolic_rat_var(param_c_name),)


def get_util_exprs(dataset_size: int, dataset_seed: int):
    train_X, train_y, train_X_np, train_y_np = get_train(dataset_seed)
    n_frames = dataset_size * 100
    sys.setrecursionlimit(n_frames * 100)

    def closurify(feature: Any) -> Any:
        # needed because python doesn't create a new binding for every iteration of a comprehension
        return (
            lambda param, x: x[feature] <= param[0],
            min(x[feature] for x in train_X) - 1,
            max(x[feature] for x in train_X) + 1,
        )

    sketches: Dict[
        str, Tuple[Callable[[Param, Tuple[rat, ...]], BoolSymbolic], rat, rat]
    ] = {
        f"{feature}": closurify(feature) for feature in [4]
    }  # in range(19)

    return {
        k: (
            util(sketch, train_X[0:n_frames], train_y[0:n_frames], param).t,
            {param_c_name: Interval(feat_min, feat_max)},
        )
        for k, (sketch, feat_min, feat_max) in sketches.items()
    }


state_util_bounds = Interval(rat(0, 1), rat(1, 1))
