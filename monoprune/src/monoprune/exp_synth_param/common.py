from monoprune.env import *
import heapq
from monoprune.interval import Interval
import time
from sortedcontainers.sortedlist import SortedList

T = TypeVar("T")

ExperimentOutputLine: TypeAlias = Tuple[
    rat,  # util lower bund
    T,  # lower bound witness state
    Optional[rat],  # util upper bound
]

ExperimentOutput = Tuple[ExperimentOutputLine[T], ...]


def online_search(
    queue_key: Callable[[Interval[rat]], Any],
    util_bounds: Callable[[T], Interval[rat]],
    expand: Callable[[T], Iterable[T]],
    initial_states: Iterable[T],
) -> Iterable[ExperimentOutputLine[T]]:
    worklist: List[Tuple[Any, Interval[rat], T]] = []
    upper_bounds: SortedList = SortedList()

    greatest_lower = None
    greatest_lower_state = None
    greatest_upper = None
    state_changed = False

    def add_state(state: T):
        nonlocal worklist, greatest_lower, greatest_lower_state, state_changed
        state_util_bounds = util_bounds(state)

        if greatest_lower is None or greatest_lower < state_util_bounds.lower:
            greatest_lower = state_util_bounds.lower
            greatest_lower_state = state
            state_changed = True

        heapq.heappush(
            worklist,
            (
                queue_key(state_util_bounds),
                state_util_bounds,
                state,
            ),
        )
        upper_bounds.add(state_util_bounds.upper)  # type: ignore

    for state in initial_states:
        add_state(state)
        if state_changed:
            assert greatest_lower is not None
            assert greatest_lower_state is not None
            yield (greatest_lower, greatest_lower_state, greatest_upper)
            state_changed = False

    while len(worklist) != 0:
        new_greatest_upper: rat = upper_bounds[-1]  # type: ignore
        if greatest_upper is None or greatest_upper > new_greatest_upper:
            greatest_upper = new_greatest_upper
            state_changed = True
        assert greatest_lower is not None
        assert greatest_lower_state is not None
        assert greatest_upper is not None

        if state_changed:
            yield (greatest_lower, greatest_lower_state, greatest_upper)
            state_changed = False

        _, popped_util_bounds, state = heapq.heappop(worklist)
        upper_bounds.remove(popped_util_bounds.upper)  # type: ignore

        for new_state in expand(state):
            add_state(new_state)
            if state_changed:
                yield (greatest_lower, greatest_lower_state, greatest_upper)
                state_changed = False

def heuristic_search_key() -> Callable[[Interval[rat]], Any]:
    unique_num = 0
    def key(state_util_bounds : Interval[rat]) -> Any:
        nonlocal unique_num
        r = (
            -state_util_bounds.upper,
            -state_util_bounds.lower,
            unique_num,
        )
        unique_num += 1
        return r

    return key

def breadth_first_search_key() -> Callable[[Interval[rat]], Any]:
    unique_num = 0
    def key(_state_util_bounds : Interval[rat]) -> Any:
        nonlocal unique_num
        r = (
            unique_num,
        )
        unique_num += 1
        return r

    return key


def ctime(g: Iterable[T]) -> Iterable[Tuple[float, T]]:
    time_tot = 0.0
    time_last = time.perf_counter()
    for x in g:
        time_tot += time.perf_counter() - time_last
        yield (time_tot, x)
        time_last = time.perf_counter()


if False:

    def heuristic(x: rat) -> Interval[rat]:
        return Interval(x, rat(10, 1))

    def expand(x: rat) -> List[rat]:
        return [x + 1]

    search_process = iter(online_search(heuristic, expand, [rat(0, 1)]))
    print(next(search_process))
