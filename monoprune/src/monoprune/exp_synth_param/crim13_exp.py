from monoprune.env import *
from monoprune.exp_synth_param.common import ctime
from monoprune.exp_synth_param.interval import interval_opt
from monoprune.exp_synth_param.dyadic import dyadic_opt
from monoprune.exp_synth_param.smt import smt_opt
from monoprune.aexpr.semantics_concrete import semantics_concrete
from monoprune.aexpr.semantics_interval import semantics_interval

from monoprune.exp_synth_param.crim13_sketch import (
    get_util_exprs,
    state_util_bounds,
)

util_exprs = get_util_exprs(40, 0)


def name_of_sketch(sketch: Any) -> Optional[str]:
    for k, (sketch2, _) in util_exprs.items():
        if sketch == sketch2:
            return k
    return None


output: Dict[str, List[Tuple[float, rat, Optional[rat]]]] = {}

procedure_interval = interval_opt(util_exprs.values())
interval_res = None
interval_output: List[Tuple[float, rat, Optional[rat]]] = []
for i, (t, (lb, lb_state, ub)) in enumerate(ctime(procedure_interval)):
    ub_disp = float("nan") if ub is None else ub
    print(
        f"{i}, {t:0.2f}, {float(lb):0.2f}, {float(ub_disp):0.2f} {name_of_sketch(lb_state[0])}"
    )
    interval_output.append((t, lb, ub))
    if ub == lb:
        interval_res = lb
        break

procedure_dyadic = dyadic_opt(util_exprs.values())
dyadic_output: List[Tuple[float, rat, Optional[rat]]] = []
for i, (t, (lb, lb_state, ub)) in enumerate(ctime(procedure_dyadic)):
    ub_disp = float("nan") if ub is None else ub
    print(
        f"{i}, {t:0.2f}, {float(lb):0.2f}, {float(ub_disp):0.2f} {name_of_sketch(lb_state[0])}"
    )
    dyadic_output.append((t, lb, ub))
    if lb == interval_res:
        break

procedure_smt = smt_opt(util_exprs.values(), state_util_bounds)
for i, (t, (lb, lb_state, ub)) in enumerate(ctime(procedure_smt)):
    sketch, assignment = lb_state
    ub_disp = float("nan") if ub is None else ub
    print(
        f"{i}, {t:0.2f}, {float(lb):0.2f}, {float(ub_disp):0.2f} {name_of_sketch(lb_state[0])}"
    )
    if assignment is not None:
        print("model:", assignment)
    if lb == ub:
        if lb > interval_res:
            print("WARNING: something is terribly wrong!")
        break

if False:
    lb_state_conc = semantics_concrete(*(d.__getitem__ for d in lb_state))[2](
        util_expr_a
    )
    print(
        semantics_concrete(
            {}.__getitem__,
            {}.__getitem__,
            {
                param_names[0]: rat.from_float(25.0),
                param_names[1]: rat.from_float(16.0),
            }.__getitem__,
        )[2](util_expr_a)
    )

    print(
        semantics_interval(
            {}.__getitem__,
            {}.__getitem__,
            {
                param_names[0]: Interval(rat.from_float(25.0), rat.from_float(25.0)),
                param_names[1]: Interval(rat.from_float(16.0), rat.from_float(16.0)),
            }.__getitem__,
        )[2](util_expr_a)
    )
