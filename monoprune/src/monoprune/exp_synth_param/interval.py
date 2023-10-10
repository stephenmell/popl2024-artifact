from monoprune.env import *
import monoprune.exp_synth_param.common as common
from monoprune.interval import Interval
from monoprune.unique import Unique, unique
from monoprune.aexpr.syntax import RatTerm
from monoprune.aexpr.semantics_interval import semantics_interval
from monoprune.aexpr.semantics_concrete import semantics_concrete

import itertools


def make_all_boolean_combinations(ks: Iterable[Unique]) -> Iterable[Dict[Unique, bool]]:
    ordered = tuple(ks)
    for bs in itertools.product((False, True), repeat=len(ordered)):
        yield {ordered[i]: b for i, b in enumerate(bs)}


def split_interval_context(
    rat_ctx: Dict[Unique, Interval[rat]]
) -> Iterable[Dict[Unique, Interval[rat]]]:
    for kb in make_all_boolean_combinations(rat_ctx.keys()):
        yield {
            k: Interval(v.lower, mid) if kb[k] else Interval(mid, v.upper)
            for k, v in rat_ctx.items()
            for mid in [(v.lower + v.upper) / 2]
        }


if False:
    tuple(
        split_interval_context(
            {
                unique("a"): Interval(rat(0, 1), rat(1, 1)),
                unique("b"): Interval(rat(0, 1), rat(1, 1)),
            }
        )
    )

IntervalState: TypeAlias = Tuple[RatTerm, Dict[Unique, Interval[rat]]]


def _state_evaluator(do_interval: bool):
    def _eval_state(state: IntervalState) -> Interval[rat]:
        midpoint: Dict[Unique, rat] = {
            k: (v.upper + v.lower) / 2 for k, v in state[1].items()
        }
        midpoint_output = semantics_concrete(
            dictfun({}), dictfun({}), dictfun(midpoint)
        )[2](state[0])

        if do_interval:
            interval_output = semantics_interval(
                dictfun({}), dictfun({}), dictfun(state[1])
            )[2](state[0])
            assert midpoint_output >= interval_output.lower
        else:
            # TODO: take initial bounds as input
            interval_output = Interval(rat(0, 1), rat(1, 1))
        return Interval(midpoint_output, interval_output.upper)

    return _eval_state


def _expand_state(state: IntervalState) -> Iterable[IntervalState]:
    for new_param_bounds in split_interval_context(state[1]):
        yield (state[0], new_param_bounds)


def heuristic_opt(
    sketches_with_bounds: Iterable[IntervalState],
) -> Iterable[common.ExperimentOutputLine[IntervalState]]:
    return common.online_search(
        common.heuristic_search_key(),
        _state_evaluator(True),
        _expand_state,
        sketches_with_bounds,
    )


def breadth_first_opt(
    sketches_with_bounds: Iterable[IntervalState],
) -> Iterable[common.ExperimentOutputLine[IntervalState]]:
    return common.online_search(
        common.breadth_first_search_key(),
        _state_evaluator(True),
        _expand_state,
        sketches_with_bounds,
    )


def dyadic2_opt(
    sketches_with_bounds: Iterable[IntervalState],
) -> Iterable[common.ExperimentOutputLine[IntervalState]]:
    return common.online_search(
        common.breadth_first_search_key(),
        _state_evaluator(False),
        _expand_state,
        sketches_with_bounds,
    )
