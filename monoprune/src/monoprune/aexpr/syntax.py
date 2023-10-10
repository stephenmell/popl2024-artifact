from typing import TypeAlias
from monoprune.env import *
from monoprune.unique import Unique

BoolTerm: TypeAlias = Union[
    Tuple[Literal["var"], Tuple[Unique]],
    Tuple[Literal["lit"], Tuple[bool]],
    Tuple[Literal["inv"], Tuple["BoolTerm"]],
    Tuple[Literal["and"], Tuple["BoolTerm", "BoolTerm"]],
    Tuple[Literal["or"], Tuple["BoolTerm", "BoolTerm"]],
    Tuple[Literal["ite"], Tuple["BoolTerm", "BoolTerm", "BoolTerm"]],
    Tuple[Literal["lt"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["le"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["gt"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["ge"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["eq"], Tuple["RatTerm", "RatTerm"]],
]

IntTerm: TypeAlias = Union[
    Tuple[Literal["var"], Tuple[Unique]],
    Tuple[Literal["lit"], Tuple[int]],
    #    Tuple[Literal["floor"], Tuple["RatTerm"]],
    #    Tuple[Literal["ceil"], Tuple["RatTerm"]],
    Tuple[Literal["neg"], Tuple["IntTerm"]],
    Tuple[Literal["add"], Tuple["IntTerm", "IntTerm"]],
    Tuple[Literal["mul"], Tuple["IntTerm", "IntTerm"]],
    Tuple[Literal["ite"], Tuple["BoolTerm", "IntTerm", "IntTerm"]],
]


RatTerm: TypeAlias = Union[
    Tuple[Literal["var"], Tuple[Unique]],
    Tuple[Literal["lit"], Tuple[rat]],
    Tuple[Literal["of_int"], Tuple["IntTerm"]],
    Tuple[Literal["neg"], Tuple["RatTerm"]],
    Tuple[Literal["add"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["mul"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["truediv"], Tuple["RatTerm", "RatTerm"]],
    Tuple[Literal["sb_right"], Tuple["RatTerm"]],
    Tuple[Literal["ite"], Tuple["BoolTerm", "RatTerm", "RatTerm"]],
]

# BoolDenotation = TypeVar("BoolDenotation")
# BoolDenotation = TypeVar("BoolDenotation")
# BoolDenotation = TypeVar("BoolDenotation")
# BoolContext : TypeAlias = Callable[[Unique], bool]
# IntContext : TypeAlias = Callable[[Unique], int]
# RatContext : TypeAlias = Callable[[Unique], rat]
# BoolEvaluator : TypeAlias = Callable[[BoolTerm], bool]
# IntEvaluator : TypeAlias = Callable[[IntTerm], int]
# RatEvaluator : TypeAlias = Callable[[RatTerm], rat]
# Semantics : TypeAlias = Callable[
#     [BoolContext, IntContext, RatContext],
#     Tuple[BoolEvaluator, IntEvaluator, RatEvaluator],
# ]