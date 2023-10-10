from monoprune.env import *
import monoprune.exp_synth_param.common as common
from monoprune.interval import Interval
from monoprune.unique import Unique
from monoprune.aexpr.syntax import RatTerm, BoolTerm
from monoprune.aexpr.semantics_concrete import semantics_concrete
from monoprune.aexpr.smt_check import smt_check, Vars, VarAssignments

SMTInput: TypeAlias = Tuple[RatTerm, Dict[Unique, Interval[rat]]]
# state components are: sketch, sketch param bounds, model witnessing lower bound, F1 bounds, last_was_sat
SMTState: TypeAlias = Tuple[
    RatTerm,
    Dict[Unique, Interval[rat]],
    Optional[Dict[Unique, rat]],
    Interval[rat],
    bool,
]
SMTOutput: TypeAlias = Tuple[RatTerm, Optional[Dict[Unique, rat]]]


def _smt_opt(
    sketches_with_bounds: Iterable[SMTInput],
    objective_bounds: Interval[rat],
) -> Iterable[common.ExperimentOutputLine[SMTState]]:
    def eval_state(state: SMTState) -> Interval[rat]:
        _sketch, _sketch_bounds, _lb_model, util_bounds, _last_was_sat = state
        return util_bounds

    def expand_state(state: SMTState) -> Iterable[SMTState]:
        sketch, sketch_bounds, lb_model, util_bounds, last_was_sat = state

        if last_was_sat:
            util_target = (util_bounds.upper + util_bounds.lower) / 2
        else:
            util_target = util_bounds.lower

        constraint_expr: BoolTerm = ("gt", (sketch, ("lit", (util_target,))))
        sat, util_target_model = smt_check(
            (frozenset(), frozenset(), frozenset(sketch_bounds.keys())), constraint_expr
        )
        if sat:
            assert util_target_model is not None
            for k, v in util_target_model[2].items():
                assert k in sketch_bounds
                assert v >= sketch_bounds[k].lower
                assert v <= sketch_bounds[k].upper

            _, _, concrete_rat = semantics_concrete(
                *(v.__getitem__ for v in util_target_model)  # type: ignore
            )
            util_concrete = concrete_rat(sketch)
            assert util_concrete > util_bounds.lower
            new_util_bounds = Interval(util_concrete, util_bounds.upper)
            new_util_lower_model = util_target_model[2]
        else:
            new_util_bounds = Interval(util_bounds.lower, util_target)
            new_util_lower_model = lb_model

        return ((sketch, sketch_bounds, new_util_lower_model, new_util_bounds, sat),)

    return common.online_search(
        common.heuristic_search_key(),
        eval_state,
        expand_state,
        (
            (sketch, sketch_bounds, None, objective_bounds, True)
            for sketch, sketch_bounds in sketches_with_bounds
        ),
    )


def smt_opt(
    sketches_with_bounds: Iterable[SMTInput],
    objective_bounds: Interval[rat],
) -> Iterable[common.ExperimentOutputLine[SMTOutput]]:
    for (
        lb,
        (sketch, _sketch_bounds, lb_model, _util_bounds, _last_was_sat),
        ub,
    ) in _smt_opt(sketches_with_bounds, objective_bounds):
        yield lb, (sketch, lb_model), ub
