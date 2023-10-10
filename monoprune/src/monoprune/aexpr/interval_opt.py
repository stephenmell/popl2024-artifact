from monoprune.env import *
from monoprune.interval import Interval
from monoprune.unique import Unique
from monoprune.aexpr.syntax import RatTerm
from monoprune.aexpr.semantics_interval import semantics_interval


def split_interval_context(
    rat_ctx: Dict[Unique, Interval[rat]]
) -> Iterable[Dict[Unique, Interval[rat]]]:
    assert len(rat_ctx) == 1

    k, v = next(iter(rat_ctx.items()))
    mid = (v.lower + v.upper) / 2

    return tuple(
        [
            {k: Interval(v.lower, mid)},
            {k: Interval(mid, v.upper)},
        ]
    )


# For future use when we want to do more than 1D
""" def split_bounds(bounds):
    return tuple(
        tuple(
            Interval((bounds[i].lower + bounds[i].upper) / 2, bounds[i].upper)
            if is_upper
            else Interval(bounds[i].lower, (bounds[i].lower + bounds[i].upper) / 2)
            for i, is_upper in enumerate(p)
        )
        for p in itertools.product((False, True), repeat=len(bounds))
    ) """


IntervalState: TypeAlias = Dict[Unique, Interval[rat]]


def interval_opt(
    rat_ctx: IntervalState,
    util_expr: RatTerm,
    tolerance: rat,
) -> Tuple[Interval[rat], IntervalState]:
    def interval_eval(state: IntervalState) -> Interval[rat]:
        return semantics_interval(dictfun({}), dictfun({}), dictfun(state))[2](
            util_expr
        )

    def util_heuristic(state: IntervalState) -> rat:
        return interval_eval(state).upper

    def terminal(state: IntervalState) -> bool:
        interval = interval_eval(state)
        return interval.upper - interval.lower <= tolerance

    res = heuristic_search(util_heuristic, split_interval_context, terminal, [rat_ctx])

    assert res is not None
    return (interval_eval(res), res)


T = TypeVar("T")


import heapq


def heuristic_search(
    util_heuristic: Callable[[T], rat],
    expand: Callable[[T], Iterable[T]],
    terminal: Callable[[T], bool],
    initial_states: Iterable[T],
) -> Optional[T]:
    unique_num = 0
    # tuples of form (util ub, unique_num, state)
    worklist: List[Tuple[rat, int, T]] = []

    def add_state(state: T):
        nonlocal unique_num, worklist
        heapq.heappush(
            worklist,
            (
                -util_heuristic(state),
                unique_num,
                state,
            ),
        )
        unique_num += 1

    for state in initial_states:
        add_state(state)

    while len(worklist) != 0:
        _, _, state = heapq.heappop(worklist)
        if terminal(state):
            return state
        else:
            for new_state in expand(state):
                add_state(new_state)
    return None


if False:

    def find_optimal_param(objective, initial_interval, tolerance):
        _unique_num = 0
        worklist = []
        best_realizer = None
        best_cost = float("inf")

        def add_node(new_param_interval):
            nonlocal _unique_num, worklist, best_realizer, best_cost
            new_cost_interval = objective(new_param_interval)
            # print(new_cost_interval, new_param_interval)
            if new_cost_interval.upper < best_cost:
                print("new best cost", best_cost)
                best_cost = new_cost_interval.upper
                best_realizer = new_param_interval
            # print("\tpushing cost bound", new_cost_interval, new_param_interval)
            heapq.heappush(
                worklist,
                (
                    (new_cost_interval.lower, new_cost_interval.upper),
                    _unique_num,
                    new_cost_interval,
                    new_param_interval,
                ),
            )
            _unique_num += 1

        add_node(initial_interval)

        while len(worklist) != 0:
            (cost_lower_bound, _), _, cost_interval, param_interval = heapq.heappop(
                worklist
            )
            global_range = best_cost - cost_lower_bound
            # print("GR", cost_lower_bound, best_cost, global_range)
            if global_range <= tolerance:
                break
            # print("popping cost bound", cost_interval, param_interval)
            for new_param_interval in split_bounds(param_interval):
                add_node(new_param_interval)

        return best_cost, best_realizer
