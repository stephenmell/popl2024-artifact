from monoprune.env import *

T = TypeVar("T")


@dataclass(frozen=True)
class Interval(Generic[T]):
    lower: T
    upper: T

    def __post_init__(self):
        assert type(self.lower) == type(self.upper)
        assert self.lower <= self.upper, f"{self.lower} > {self.upper}"  # type: ignore


def interval_ite(c: Interval[bool], a: Interval[T], b: Interval[T]) -> Interval[T]:
    if c.lower == True:
        return a
    elif c.upper == False:
        return b
    else:
        return Interval(min(a.lower, b.lower), max(a.upper, b.upper))  # type: ignore


def interval_lt(or_eq: bool, a: Interval[rat], b: Interval[rat]) -> Interval[bool]:
    if or_eq:
        some_lt = a.lower <= b.upper
        all_lt = a.upper <= b.lower
    else:
        some_lt = a.lower < b.upper
        all_lt = a.upper < b.lower
    return Interval(all_lt, some_lt)


def interval_and(a: Interval[bool], b: Interval[bool]) -> Interval[bool]:
    return Interval(a.lower and b.lower, a.upper and b.upper)


def interval_neg(a: Interval[T]) -> Interval[T]:
    return Interval(-a.upper, -a.lower)  # type: ignore


def interval_add(a: Interval[T], b: Interval[T]) -> Interval[T]:
    return Interval(a.lower + b.lower, a.upper + b.upper)  # type: ignore


def interval_mul(a: Interval[T], b: Interval[T]) -> Interval[T]:
    w = a.lower * b.lower  # type: ignore
    x = a.lower * b.upper  # type: ignore
    y = a.upper * b.lower  # type: ignore
    z = a.upper * b.upper  # type: ignore
    return Interval(min(w, x, y, z), max(w, x, y, z))  # type: ignore


def interval_truediv(a: Interval[rat], b: Interval[rat]) -> Interval[rat]:
    assert b.lower > 0
    return interval_mul(a, Interval(1 / b.upper, 1 / b.lower))

def interval_sb_right(a: Interval[rat]) -> Interval[rat]:
    assert a.lower >= 0
    return Interval(a.lower / (1 + a.lower), a.upper / (1 + a.upper))