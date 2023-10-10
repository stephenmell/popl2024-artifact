from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm
from monoprune.aexpr.semantics_z3 import semantics_z3
import z3

Vars: TypeAlias = Tuple[
    FrozenSet[Unique],
    FrozenSet[Unique],
    FrozenSet[Unique],
]
VarAssignments: TypeAlias = Tuple[
    Dict[Unique, bool],
    Dict[Unique, int],
    Dict[Unique, rat],
]


def smt_check(
    vars: Vars,
    expr: BoolTerm,
) -> Tuple[bool, Optional[VarAssignments]]:
    bool_vars, int_vars, rat_vars = vars
    smt_vars = {}
    for bool_var in bool_vars:
        smt_vars[bool_var] = z3.Bool(f"{bool_var}")
    for int_var in int_vars:
        smt_vars[int_var] = z3.Int(f"{int_var}")
    for rat_var in rat_vars:
        smt_vars[rat_var] = z3.Real(f"{rat_var}")

    eval_bool, _, _ = semantics_z3(
        lambda v: smt_vars[v], lambda v: smt_vars[v], lambda v: smt_vars[v]
    )
    smt_expr = eval_bool(expr)
    solver = z3.Solver()
    solver.add(smt_expr)

    result = solver.check()
    if result == z3.unsat:
        succ = False
        smt_model = None
    elif result == z3.sat:
        succ = True
        smt_model = solver.model()
    elif result == z3.unknown:
        print("smt failed")
        print(solver.model())
        assert False
    else:
        assert False

    if smt_model is not None:
        bool_model = {k: bool(smt_model[smt_vars[k]]) for k in bool_vars}
        int_model = {k: smt_model[smt_vars[k]].as_long() for k in int_vars}
        rat_model = {k: smt_model[smt_vars[k]].as_fraction() for k in rat_vars}
        return succ, (bool_model, int_model, rat_model)
    else:
        return succ, None


if False:
    from monoprune.aexpr.symbolic_syntax import *
    from monoprune.unique import unique

    x_name = unique("x")
    x = symbolic_rat_var(x_name)
    expr = x == x + 1
    smt_check(set(), set(), {x_name}, expr.t)
