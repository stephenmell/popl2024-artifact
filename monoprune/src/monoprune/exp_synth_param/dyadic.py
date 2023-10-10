from monoprune.env import *
import monoprune.exp_synth_param.common as common
from monoprune.interval import Interval
from monoprune.unique import Unique, unique
from monoprune.aexpr.syntax import RatTerm
from monoprune.aexpr.semantics_concrete import semantics_concrete


def _gen_numers_for_denom(denom: int, dim: int) -> Iterable[Tuple[int, ...]]:
    if dim == 0:
        yield tuple()  # type: ignore
    else:
        num = 0
        while num <= denom:
            for l in _gen_numers_for_denom(denom, dim - 1):
                yield (num,) + l
            num += 1


def _dyadic_rats(dim: int) -> Iterable[Tuple[rat, ...]]:
    denom = 1

    yield (rat(0, 1),) * dim

    while True:
        for numers in _gen_numers_for_denom(denom, dim):
            if not all(c % 2 == 0 for c in numers):
                yield tuple(rat(c, denom) for c in numers)
        denom *= 2


if False:
    tuple(_gen_numers_for_denom(2, 2))
    import itertools

    tuple(itertools.islice(_dyadic_rats(2), 10))


def dyadic_rats(
    rat_ctx: Dict[Unique, Interval[rat]],
) -> Iterable[Dict[Unique, rat]]:
    ordered_ks = tuple(rat_ctx.keys())
    dyadic = _dyadic_rats(len(ordered_ks))
    for v in dyadic:
        yield {
            k: v[i] * rat_ctx[k].upper + (1 - v[i]) * rat_ctx[k].lower
            for i, k in enumerate(ordered_ks)
        }


if False:
    import itertools
    import plotly.express as px

    x = unique("x")
    y = unique("y")
    points = tuple(
        tuple([float(d[x]), float(d[y])])
        for d in itertools.islice(
            dyadic_rats(
                {
                    x: Interval(rat(0, 1), rat(1, 1)),
                    y: Interval(rat(0, 1), rat(1, 1)),
                }
            ),
            1000,
        )
    )
    px.scatter(x=[p[0] for p in points], y=[p[1] for p in points])

SketchWithBounds: TypeAlias = Tuple[RatTerm, Dict[Unique, Interval[rat]]]
SketchWithParams: TypeAlias = Tuple[RatTerm, Dict[Unique, rat]]

T = TypeVar("T")
S = TypeVar("S")
from itertools import cycle, islice


def roundrobin(iterables: Tuple[Iterable[T], ...]) -> Iterable[T]:
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def roundreobin_over_dict(d: Dict[S, Iterable[T]]) -> Iterable[Tuple[S, T]]:
    def augment(k: S, g: Iterable[T]) -> Iterable[Tuple[S, T]]:
        for v in g:
            yield (k, v)

    return roundrobin(tuple(augment(k, g) for k, g in d.items()))


def dyadic_opt(
    sketches_with_bounds: Iterable[SketchWithBounds],
) -> Iterable[common.ExperimentOutputLine[SketchWithParams]]:
    util_lower = None
    util_lower_model = None

    for k, x in roundreobin_over_dict(
        {k: dyadic_rats(v) for k, v in sketches_with_bounds}
    ):
        x_util = semantics_concrete(dictfun({}), dictfun({}), dictfun(x))[2](k)

        if util_lower is None or util_lower < x_util:
            util_lower = x_util
            util_lower_model = (k, x)

            yield (
                util_lower,
                util_lower_model,
                None,
            )
