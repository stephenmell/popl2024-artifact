from monoprune.env import *

import monoprune.sb.smt_exp_crim as smt_exp_crim
import monoprune.sb.smt_exp_emergency_quivr as smt_exp_emergency_quivr

from jaxtyping import Float, Bool, Int, Shaped
import traceback
from torch import Tensor
import torch
from monoprune.crim13_poly_dsl.syntax import *
from monoprune.crim13_poly_dsl.semantics import *
import monoprune.crim13_poly_dsl.enumeration as enumeration
from monoprune.exp_synth_param.common import ctime
from monoprune.exp_torch.common import *

import sys

torch.set_num_threads(1)

if "get_ipython" in globals():
    # metadataset = "crim"
    # dataset_name = "crim"
    # dataset_task = "a"
    metadataset = "emergency_quivr"
    dataset_name = "maritime_surveillance"
    dataset_task = "a"

    dataset_size_str = "30"
    dataset_seed_str = "0"
    arg_approach_str = "heuristic"
    arg_timeout_secs_str = "600"
    print_to_files = True
    print_expand_stdout = False
else:
    (
        _,
        metadataset,
        dataset_name,
        dataset_task,
        arg_approach_str,
        arg_timeout_secs_str,
        dataset_size_str,
        dataset_seed_str,
    ) = sys.argv
    print_to_files = True
    print_expand_stdout = False
arg_timeout_secs = float(arg_timeout_secs_str)
arg_train_util = "f1"
dataset_size = int(dataset_size_str)
dataset_seed = int(dataset_seed_str)

output_path = lambda s: f"output_exp_smt/{dataset_name}_{dataset_task}_{dataset_size_str}_{dataset_seed_str}_{arg_approach_str}_{arg_train_util}_{arg_timeout_secs_str}_{s}"  # type: ignore
if print_to_files:
    output_bound_fd = open(output_path("bound"), "w")
    output_expand_fd = open(output_path("expand"), "w")
else:
    output_bound_fd = None
    output_expand_fd = None

if metadataset == "marisurv":
    assert False
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = smt_exp_marisurv.get_experiment(dataset_name, dataset_task)
elif metadataset == "emergency_quivr":
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = smt_exp_emergency_quivr.get_experiment(
        dataset_name, dataset_task, dataset_size, dataset_seed
    )
elif metadataset == "crim":
    (
        sketches_with_bounds,
        util_f1,
        util_f1_interval,
        util_bce,
        util_bce_interval,
    ) = smt_exp_crim.get_experiment(dataset_name, dataset_task)
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
        # print(i)
        global _dbg_considered_states
        _dbg_considered_states.append(i)
        o = inner(i)

        global _dbg_iter
        _dbg_iter += 1

        # if i[0] == ("map", (((), (((),), (0,))),)):
        t_elap = time.perf_counter() - _dbg_t_init
        # if _dbg_iter % 1000 == 0:
        if True:
            if output_expand_fd is not None:
                output_expand_fd.write(f"{_dbg_iter},{t_elap}\n")
                output_expand_fd.write(f"\t {i[0]}\n")
                output_expand_fd.write(f"\t {i[1]} {i[2]}\n")
                output_expand_fd.write(f"\t {o}\n")
                output_expand_fd.flush()
        if print_expand_stdout:
            print(f"{_dbg_iter},{t_elap}")
            print(f"\t {i[0]}")
            print(f"\t {i[1]} {i[2]}")
            print(f"\t {o}")

        if t_elap > arg_timeout_secs:
            assert False, "TIMEOUT"
        if False:
            # a + b = 0
            # a is dim 0, b is dim 1
            spans_line_lb = i[1][0] + i[1][1]
            spans_line_ub = i[2][0] + i[2][1]
            if (spans_line_lb < 0) and (spans_line_ub > 0):
                print("!")
            spans_line2_lb = i[1][0] - i[2][1]
            spans_line2_ub = i[2][0] - i[1][1]
            if (spans_line2_lb < 0) and (spans_line2_ub > 0):
                print("?")
            # dim_spans_zero = (i[1] < 0) & (i[2] > 0)
            # if dim_spans_zero.any():
            #     print("!")
        if False:
            eps = 0.00000001
            lb_belowz = (i[1] < 0).all()
            lb_beloweqz = (i[1] <= 0).all()
            ub_abovez = (i[2] > 0).all()
            ub_aboveeqz = (i[2] >= 0).all()
            if lb_belowz and ub_abovez:
                print("!")
            if lb_belowz and not ub_aboveeqz:
                if not o.upper == 0.0:
                    print("WARN negparam", i, o)
            if ub_abovez and not lb_beloweqz:
                print("WARN posparam", i, o)
        # if lb_beloweqz or ub_aboveeqz:
        #     lb_str = "I" if lb_belowz else "T" if lb_beloweqz else "O"
        #     ub_str = "I" if ub_abovez else "T" if ub_aboveeqz else "O"
        #     print(lb_str, ub_str)
        # if lb_belowz and ub_abovez:
        #     print("spans zero")
        # else:
        #     print("touches zero")
        # print(i)
        # print(o)

        # print("STATE_EVAL")
        # print(i)
        # print(o)
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
        # print("Unexpected terminal state:")
        # print(state[0])
        # print(state[1], state[1].shape)
        # print(state[2], state[2].shape)
        return ()
    return expand_state(state)


float(rat(1644869, 67108864))

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

if False:
    f1_dyadic2_opt = online_search(
        breadth_first_search_key(),
        dbg_state_evaluator(util_f1, None),
        dbg_expand_state,
        sketches_with_bounds,
    )

    crossent_heuristic_opt = online_search(
        heuristic_search_key(),
        dbg_state_evaluator(util_bce, util_bce_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )

    crossent_breadth_first_opt = online_search(
        breadth_first_search_key(),
        dbg_state_evaluator(util_bce, util_bce_interval),
        dbg_expand_state,
        sketches_with_bounds,
    )

    crossent_dyadic2_opt = online_search(
        breadth_first_search_key(),
        dbg_state_evaluator(util_bce, None),
        dbg_expand_state,
        sketches_with_bounds,
    )

if arg_approach_str == "heuristic":
    procedure = heuristic_opt
elif arg_approach_str == "bfs":
    procedure = breadth_first_opt
else:
    assert False, arg_approach_str
# procedure = breadth_first_opt(sketches_with_bounds)
# optimal_util = lb
# procedure = dyadic2_opt(sketches_with_bounds)
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
