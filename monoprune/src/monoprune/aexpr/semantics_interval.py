from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm  # type: ignore
from monoprune.interval import *


def _empty_dict_fun(k: Any) -> Any:
    assert False


def semantics_interval(
    ctx_bool: Callable[[Unique], Interval[bool]] = _empty_dict_fun,
    ctx_int: Callable[[Unique], Interval[int]] = _empty_dict_fun,
    ctx_rat: Callable[[Unique], Interval[rat]] = _empty_dict_fun,
) -> Tuple[
    Callable[[BoolTerm], Interval[bool]],
    Callable[[IntTerm], Interval[int]],
    Callable[[RatTerm], Interval[rat]],
]:
    def eval_bool(e: BoolTerm) -> Interval[bool]:
        match e:
            case "var", (var,):
                ret = ctx_bool(var)
            case "lit", (lit,):
                ret = Interval(lit, lit)
            case "inv", (a,):
                v_a = eval_bool(a)
                ret = Interval(not v_a.upper, not v_a.lower)
            case "and", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = interval_and(v_a, v_b)
            case "or", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = Interval(v_a.lower or v_b.lower, v_a.upper or v_b.upper)
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = interval_ite(v_c, v_a, v_b)
            case "lt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_lt(False, v_a, v_b)
            case "le", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_lt(True, v_a, v_b)
            case "gt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_lt(False, v_b, v_a)
            case "ge", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_lt(True, v_b, v_a)
            case "eq", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_and(
                    interval_lt(True, v_a, v_b), interval_lt(True, v_b, v_a)
                )
        assert isinstance(ret.lower, bool)
        return ret

    def eval_int(e: IntTerm) -> Interval[int]:
        match e:
            case "var", (var,):
                ret = ctx_int(var)
            case "lit", (lit,):
                ret = Interval(lit, lit)
            case "neg", (a,):
                v_a = eval_int(a)
                ret = interval_neg(v_a)
            case "add", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = interval_add(v_a, v_b)
            case "mul", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = interval_mul(v_a, v_b)
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = interval_ite(v_c, v_a, v_b)
        assert isinstance(ret.lower, int)
        return ret

    def eval_rat(e: RatTerm) -> Interval[rat]:
        match e:
            case "var", (var,):
                ret = ctx_rat(var)
            case "lit", (lit,):
                ret = Interval(lit, lit)
            case "of_int", (a,):
                v_a = eval_int(a)
                ret = Interval(rat(v_a.lower, 1), rat(v_a.upper, 1))
            case "neg", (a,):
                v_a = eval_rat(a)
                ret = interval_neg(v_a)
            case "add", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_add(v_a, v_b)
            case "mul", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_mul(v_a, v_b)
            case "truediv", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_truediv(v_a, v_b)
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = interval_ite(v_c, v_a, v_b)
            case "sb_right", (a,):
                v_a = eval_rat(a)
                ret = interval_sb_right(v_a)
        assert isinstance(ret.lower, rat)
        return ret

    return eval_bool, eval_int, eval_rat


if False:
    from monoprune.unique import unique
    from monoprune.aexpr.symbolic_syntax import *

    x = unique("x")
    eb: BoolTerm = (
        "and",
        (
            ("lit", (True,)),
            ("var", (x,)),
        ),
    )

    print(
        semantics_interval(
            lambda _: Interval(False, True), lambda _: None, lambda _: None
        )[0](eb)
    )

    xs = symbolic_rat_var(x)
    er: RatTerm = (xs * 2 + 3).t

    print(
        semantics_interval(
            lambda _: None, lambda _: None, lambda _: Interval(rat(1, 1), rat(2, 1))
        )[2](er)
    )

    for i in range(10000):
        semantics_interval_sample(
            lambda _: None, lambda _: None, lambda _: Interval(rat(1, 1), rat(2, 1))
        )[2](er)
