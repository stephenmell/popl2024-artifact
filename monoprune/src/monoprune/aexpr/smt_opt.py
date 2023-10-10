from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm
from monoprune.aexpr.smt_check import smt_check, Vars, VarAssignments
from monoprune.aexpr.semantics_concrete import semantics_concrete


def smt_opt(
    vars: Vars,
    util_expr: RatTerm,
    util_lower_initial: rat,
    util_upper_initial: rat,
    tolerance: rat,
    util_target: Optional[rat] = None,
) -> Tuple[rat, rat, Optional[VarAssignments]]:
    util_lower: rat = util_lower_initial
    util_upper: rat = util_upper_initial
    util_model: Optional[VarAssignments] = None

    while util_upper - util_lower > tolerance:
        util_target = (util_lower + util_upper) / 2
        constraint_expr: BoolTerm = ("ge", (util_expr, ("lit", (util_target,))))
        succ, util_target_model = smt_check(vars, constraint_expr)
        if not succ:
            util_upper = util_target
        else:
            util_lower = util_target
            util_model = util_target_model
    
        if util_model is not None: # sanity check
            _, _, concrete_rat = semantics_concrete(*(v.__getitem__ for v in util_model))
            util_concrete = concrete_rat(util_expr)
            assert util_concrete >= util_lower, (util_model, util_concrete, util_lower)
            assert util_concrete <= util_upper, (util_model, util_concrete, util_upper)
            if util_concrete >= util_target:
                break
    return util_lower, util_upper, util_model


if False:
    from monoprune.aexpr.symbolic_syntax import *
    from monoprune.unique import unique

    x_name = unique("x")
    x = symbolic_rat_var(x_name)
    expr = -(x + -2) * x
    print(
        smt_opt(
            (frozenset(), frozenset(), frozenset({x_name})),
            expr.t,
            rat(-10, 1),
            rat(10, 1),
            rat(1, 10),
        )
    )
