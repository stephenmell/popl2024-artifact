from monoprune.env import *

import monoprune.sb.bfs_exp_quivr as bfs_exp_quivr
import monoprune.sb.bfs_exp_crim as bfs_exp_crim

from jaxtyping import Float, Bool, Int, Shaped
import traceback
from torch import Tensor
import torch
from monoprune.crim13_poly_dsl.syntax import *
from monoprune.crim13_poly_dsl.semantics import *
from monoprune.exp_synth_param.common import ctime
from monoprune.exp_torch.common import *

import sys
torch.set_printoptions(precision=16)

(
    _,
    metadataset,
    dataset_name,
    dataset_task,
    arg_approach_str,
    arg_train_util,
    arg_timeout_secs_str,
) = sys.argv
print_to_files = True
print_expand_stdout = False
arg_timeout_secs = float(arg_timeout_secs_str)

output_path = lambda s: f"output_exp_bfs/{dataset_name}_{dataset_task}_{arg_approach_str}_{arg_train_util}_{arg_timeout_secs_str}_{s}"  # type: ignore
if print_to_files:
    output_bound_fd = open(output_path("bound"), "w")
    output_expand_fd = open(output_path("expand"), "w")
else:
    output_bound_fd = None
    output_expand_fd = None

if metadataset == "quivr":
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = bfs_exp_quivr.get_experiment(dataset_name, dataset_task)
elif metadataset == "crim":
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = bfs_exp_crim.get_experiment(dataset_name, dataset_task)
elif metadataset == "crim_all":
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = bfs_exp_crim.get_experiment("crim", dataset_task, None)
else:
    assert False, metadataset

_dbg_iter = 0
_dbg_considered_states = []
import time

_dbg_t_init = time.perf_counter()

print("TOTAL SKETCHES:", len(sketches_with_bounds))


def dbg_state_evaluator(
    concrete_util: ConcreteUtil[T],
    interval_util: Optional[IntervalUtil[T]],
) -> Callable[[IntervalState[T]], Interval[rat]]:
    inner = state_evaluator(concrete_util, interval_util)

    def ret(i: IntervalState[T]) -> Interval[rat]:
        global _dbg_considered_states
        _dbg_considered_states.append(i)
        o = inner(i)

        global _dbg_iter
        _dbg_iter += 1

        t_elap = time.perf_counter() - _dbg_t_init
        if output_expand_fd is not None:
            output_expand_fd.write(f"{_dbg_iter},{t_elap}\n")
            output_expand_fd.write(f"\t {i[0]}\n")
            output_expand_fd.write(f"\t {i[1]} {i[2]}\n")
            output_expand_fd.write(f"\t {o}\n")
            output_expand_fd.flush()
        else:
            if print_expand_stdout:
                print(f"{_dbg_iter},{t_elap}")
                print(f"\t {i[0]}")
                print(f"\t {i[1]} {i[2]}")
                print(f"\t {o}")

        if t_elap > arg_timeout_secs:
            assert False, "TIMEOUT"
        return o

    return ret


def parameter_box_is_terminal(
    parameters_lower: Float[Tensor, "param"],
    parameters_upper: Float[Tensor, "param"],
):
    if parameters_lower.shape[0] == 0:
        return True
    return torch.min(parameters_upper - parameters_lower) <= 0.00001


def dbg_expand_state(state: IntervalState[T]) -> Iterable[IntervalState[T]]:
    if parameter_box_is_terminal(state[1], state[2]):
        return ()
    return expand_state(state)


if arg_train_util == "f1":
    heuristic_opt = online_search(
        heuristic_search_key(),
        dbg_state_evaluator(util_f1, util_f1_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )

    breadth_first_opt = online_search(
        breadth_first_search_key(),
        dbg_state_evaluator(util_f1, util_f1_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )
elif arg_train_util == "bce":
    heuristic_opt = online_search(
        heuristic_search_key(),
        dbg_state_evaluator(util_bce, util_bce_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )

    breadth_first_opt = online_search(
        breadth_first_search_key(),
        dbg_state_evaluator(util_bce, util_bce_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )
else:
    assert False, arg_train_util

if arg_approach_str == "heuristic":
    procedure = heuristic_opt
elif arg_approach_str == "bfs":
    procedure = breadth_first_opt
else:
    assert False, arg_approach_str
try:
    for i, (t, (lb, lb_state, ub)) in enumerate(ctime(procedure)):
        ub_disp = float("nan") if ub is None else ub
        ub_float = float(ub) if ub is not None else float("nan")
        midpoint: Float[Tensor, "p"] = (lb_state[1] + lb_state[2]) / 2
        lb_f1 = util_f1(lb_state[0], midpoint)
        line = f"{i},{t:0.2f},{float(lb):0.2f},{float(ub_disp):0.2f},{float(lb_f1):0.2f},{lb_state}"
        if output_bound_fd is not None:
            output_bound_fd.write(f"{line}\n")
            output_bound_fd.flush()
        else:
            print(line)
        if lb == ub:
            # if lb == optimal_util:
            break
except Exception as e:
    t_elap = time.perf_counter() - _dbg_t_init
    if output_expand_fd is not None:
        output_expand_fd.write(f"err {t_elap}\n")
        output_expand_fd.write(traceback.format_exc())
        output_expand_fd.flush()
    else:
        print(f"err {t_elap}")
        print(traceback.format_exc())
    raise e
