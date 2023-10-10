from monoprune.env import *
from monoprune.exp_synth_param.interval import (
    heuristic_opt,
    breadth_first_opt,
    dyadic2_opt,
)
from monoprune.exp_synth_param.dyadic import dyadic_opt
from monoprune.exp_synth_param.smt import smt_opt
import monoprune.exp_synth_param.common as common
from monoprune.exp_synth_param.common import ctime
import pandas as pd

import sys

_, method, dataset_name, dataset_size_str, dataset_seed_str = sys.argv
dataset_size = int(dataset_size_str)
dataset_seed = int(dataset_seed_str)


def out_path(
    method: str = method,
    dataset_name: str = dataset_name,
    dataset_size: int = dataset_size,
    dataset_seed: int = dataset_seed,
):
    return f"output_exp_smt/{method}_{dataset_name}_{dataset_size}_{dataset_seed}"


def get_perfect_bound(method: str = "interval") -> float:
    final = pd.read_csv(out_path(method=method), header=None).iloc[-1]
    if not final[2] == final[3]:
        print("interval didn't terminat...")
    return final[3]


match dataset_name:
    case "crim13":
        from monoprune.exp_synth_param.crim13_sketch import (
            get_util_exprs,
            state_util_bounds,
        )
    case "emergency_quivr":
        from monoprune.exp_synth_param.emergency_quivr_sketch import (
            get_util_exprs,
            state_util_bounds,
        )

    case _:
        assert False, dataset_name

util_exprs = get_util_exprs(dataset_size, dataset_seed)


def name_of_sketch(sketch: Any) -> Optional[str]:
    for k, (sketch2, _) in util_exprs.items():
        if sketch == sketch2:
            return k
    return None


output: Dict[str, List[Tuple[float, rat, Optional[rat]]]] = {}


def stopping_criterion_perfect(lb: rat, ub: Optional[rat]) -> bool:
    return lb == ub


from monoprune.interval import Interval


def format_model(m):
    if m is None:
        return "-"

    def f(v):
        if isinstance(v, Interval):
            l = float(v.lower)
            u = float(v.upper)
            return f"[{l:0.2f}, {u:0.2f}]"
        elif isinstance(v, rat):
            return f"{float(v):0.2f}"
        else:
            assert False, v

    ret = ", ".join(f"{k.name}: {f(v)}" for k, v in m.items())
    return "{" + ret + "}"


match method:
    case "interval":
        procedure: Iterable[common.ExperimentOutputLine[Any]] = heuristic_opt(
            util_exprs.values()
        )
        stopping_criterion = stopping_criterion_perfect
    case "bfs":
        procedure: Iterable[common.ExperimentOutputLine[Any]] = breadth_first_opt(
            util_exprs.values()
        )
        stopping_criterion = stopping_criterion_perfect
    case "d2":
        procedure: Iterable[common.ExperimentOutputLine[Any]] = dyadic2_opt(
            util_exprs.values()
        )
        best_util = get_perfect_bound()

        def stopping_criterion(lb: rat, ub: Optional[rat]) -> bool:
            return float(lb) == best_util

    case "dyadic":
        procedure = dyadic_opt(util_exprs.values())
        best_util = get_perfect_bound()

        def stopping_criterion(lb: rat, ub: Optional[rat]) -> bool:
            return float(lb) == best_util

    case "smt":
        procedure = smt_opt(util_exprs.values(), state_util_bounds)
        stopping_criterion = stopping_criterion_perfect
    case _:
        assert False, method

output: List[Tuple[float, rat, Optional[rat]]] = []

with open(out_path(), "w") as out_file:
    for i, (t, (lb, lb_state, ub)) in enumerate(ctime(procedure)):
        ub_disp = float("nan") if ub is None else ub
        ub_float = float(ub) if ub is not None else float("nan")
        out_file.write(
            f"{i},{t},{float(lb)},{ub_float},{name_of_sketch(lb_state[0])}\n"
        )
        out_file.flush()
        print(
            f"{i}, {t:0.2f}, {float(lb):0.2f}, {float(ub_disp):0.2f} {name_of_sketch(lb_state[0])} {format_model(lb_state[1])}"
        )
        if stopping_criterion(lb, ub):
            break
