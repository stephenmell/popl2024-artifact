from typing import (
    Union as Union,
    Tuple as Tuple,
    Literal as Literal,
    TypeAlias as TypeAlias,
    Any as Any,
    Callable as Callable,
    TypeVar as TypeVar,
    Optional as Optional,
    FrozenSet as FrozenSet,
    MutableSet as MutableSet,
    Iterable as Iterable,
    List as List,
    Dict as Dict,
    NewType as NewType,
    Generic as Generic,
    Protocol as Protocol,
    Sequence as Sequence
)

from dataclasses import dataclass as dataclass
from fractions import Fraction as rat # type: ignore

T = TypeVar("T")

def ite(c: Any, t: T, f: T) -> T:
    if hasattr(c, "__ite__"):
        ret = c.__ite__(t, f)
        if ret is not NotImplemented:
            return ret
    elif isinstance(c, bool):
        return t if c else f

    raise NotImplementedError()

_K = TypeVar("_K")
_V = TypeVar("_V")
def dictfun(d : Dict[_K, _V]) -> Callable[[_K], _V]:
    return lambda k: d[k]