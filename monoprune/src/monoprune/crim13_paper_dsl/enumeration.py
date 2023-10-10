from monoprune.env import *

from monoprune.crim13_paper_dsl.syntax import grammar
from monoprune.type_grammar_tools import *
from monoprune.search_graph import *


def enumerate_to_depth_multi(
    cfg: Cfg, depth: int, tys: Tuple[CfgTypeName, ...]
) -> Iterable[Tuple[CfgExpr, ...]]:
    if len(tys) == 0:
        yield ()
    else:
        for hd in enumerate_to_depth(cfg, depth, tys[0]):
            for tl in enumerate_to_depth_multi(cfg, depth, tys[1:]):
                yield (hd,) + tl


def enumerate_to_depth(cfg: Cfg, depth: int, ty: CfgTypeName) -> Iterable[CfgExpr]:
    if depth > 0:
        for prod_name, prod_arg_tys in cfg[ty].items():
            for prod_args in enumerate_to_depth_multi(cfg, depth - 1, prod_arg_tys):
                yield (prod_name, prod_args)


def expr_size(partial_expr: CfgExprPartial) -> int:
    match partial_expr:
        case (None, _ty):
            return 1
        case _op_id, op_args:
            return 1 + sum(expr_size(x) for x in op_args)


def expr_complete(partial_expr: CfgExprPartial) -> bool:
    match partial_expr:
        case (None, _ty):
            return False
        case _op_id, op_args:
            return all(expr_complete(x) for x in op_args)


import itertools


def printstream(it):
    for x in it:
        print(x)
        yield x


if False:
    tuple(
        itertools.islice(
            filter(
                expr_complete,
                printstream(
                    breadth_first_search(cfg_search_graph(grammar, "CrimExprAtomAtom"))
                ),
            ),
            25,
        )
    )

l = tuple(
    filter(
        expr_complete,
        # printstream(
        breadth_first_while(
            cfg_search_graph(grammar, "CrimExprAtomAtom"),
            lambda x: expr_size(x) <= 7,
        )
        # ),
    )
)
# for x in l:
#     print(x)
print(len(l))
