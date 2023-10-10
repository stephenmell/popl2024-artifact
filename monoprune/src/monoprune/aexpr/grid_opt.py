from monoprune.env import *
from monoprune.interval import Interval
from monoprune.unique import Unique
from monoprune.aexpr.syntax import RatTerm
from monoprune.aexpr.semantics_concrete import semantics_concrete
import math


def split_interval_context(
    rat_ctx: Dict[Unique, Interval[rat]],
    rat_ctx_delta: Dict[Unique, rat],
) -> Iterable[Dict[Unique, rat]]:
    assert len(rat_ctx) == 1

    k, v = next(iter(rat_ctx.items()))
    k2, delta = next(iter(rat_ctx_delta.items()))
    assert k == k2

    n_samples = math.ceil((v.upper - v.lower) / delta)
    samples = [i * delta + v.lower for i in range(n_samples)]

    return tuple([{k: x} for x in samples])


def grid_opt(
    rat_ctx: Dict[Unique, Interval[rat]],
    util_expr: RatTerm,
    rat_ctx_delta: Dict[Unique, rat],
) -> Tuple[rat, Dict[Unique, rat]]:
    best_input = None
    best_util = None
    for input in split_interval_context(rat_ctx, rat_ctx_delta):
        input_util = semantics_concrete(dictfun({}), dictfun({}), dictfun(input))[2](
            util_expr
        )
        if best_util is None or input_util > best_util:
            best_util = input_util
            best_input = input
    assert best_util is not None
    assert best_input is not None
    return best_util, best_input


def _dyadic_rats() -> Iterable[rat]:
    yield rat(0, 1)

    denom = 1

    while True:
        num = 1
        while num <= denom:
            yield rat(num, denom)
            num += 2
        denom *= 2


def dyadic_rats(interval: Interval[rat]) -> Iterable[rat]:
    width = interval.upper - interval.lower
    inner = _dyadic_rats()

    for r in inner:
        yield r * width + interval.lower


def dyadic_sequence_from_context(
    rat_ctx: Dict[Unique, Interval[rat]],
) -> Iterable[Dict[Unique, rat]]:
    assert len(rat_ctx) == 1

    k, v = next(iter(rat_ctx.items()))

    for r in dyadic_rats(v):
        yield {k: r}


if False:
    import itertools

    tuple(itertools.islice(dyadic_rats(Interval(rat(0, 1), rat(7, 1))), 10))


def dyadic_opt(
    rat_ctx: Dict[Unique, Interval[rat]],
    util_expr: RatTerm,
    target_util: rat,
) -> Dict[Unique, rat]:
    assert len(rat_ctx) == 1
    
    for i, input in enumerate(dyadic_sequence_from_context(rat_ctx)):
        input_util = semantics_concrete(dictfun({}), dictfun({}), dictfun(input))[2](
            util_expr
        )
        if input_util >= target_util:
            print("Found on iter", i)
            return input
    assert False