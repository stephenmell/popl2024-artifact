from monoprune.env import *
from monoprune.unique import Unique
from monoprune.aexpr.syntax import BoolTerm, IntTerm, RatTerm  # type: ignore
import z3  # type: ignore


def _empty_dict_fun(k: Any) -> Any:
    assert False


def semantics_z3(
    ctx_bool: Callable[[Unique], z3.BoolRef] = _empty_dict_fun,
    ctx_int: Callable[[Unique], z3.IntNumRef] = _empty_dict_fun,
    ctx_rat: Callable[[Unique], z3.RatNumRef] = _empty_dict_fun,
) -> Tuple[
    Callable[[BoolTerm], z3.BoolRef],
    Callable[[IntTerm], z3.IntNumRef],
    Callable[[RatTerm], z3.RatNumRef],
]:
    def eval_bool(e: BoolTerm) -> z3.BoolRef:
        match e:
            case "var", (var,):
                ret = ctx_bool(var)
            case "lit", (lit,):
                ret = z3.BoolVal(lit)
            case "inv", (a,):
                v_a = eval_bool(a)
                ret = z3.Not(v_a)
            case "and", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = z3.And(v_a, v_b)
            case "or", (a, b):
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = z3.Or(v_a, v_b)
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_bool(a)
                v_b = eval_bool(b)
                ret = z3.If(v_c, v_a, v_b)
            case "lt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a < v_b
            case "le", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a <= v_b
            case "gt", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a > v_b
            case "ge", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a >= v_b
            case "eq", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a == v_b
        assert isinstance(ret, z3.BoolRef), (type(ret).__name__, ret)
        return ret

    def eval_int(e: IntTerm) -> z3.IntNumRef:
        match e:
            case "var", (var,):
                ret = ctx_int(var)
            case "lit", (lit,):
                ret = z3.IntVal(lit)
            case "neg", (a,):
                v_a = eval_int(a)
                ret = -v_a
            case "add", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = v_a + v_b
            case "mul", (a, b):
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = v_a * v_b
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_int(a)
                v_b = eval_int(b)
                ret = z3.If(v_c, v_a, v_b)
        assert isinstance(ret, z3.ArithRef), (type(ret).__name__, ret)
        return ret

    def eval_rat(e: RatTerm) -> z3.RatNumRef:
        match e:
            case "var", (var,):
                ret = ctx_rat(var)
            case "lit", (lit,):
                ret = z3.RatVal(lit.numerator, lit.denominator)
            case "of_int", (a,):
                v_a = eval_int(a)
                ret = z3.ToReal(v_a)
            case "neg", (a,):
                v_a = eval_rat(a)
                ret = -v_a
            case "add", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a + v_b
            case "mul", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a * v_b
            case "truediv", (a, b):
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = v_a / v_b
            case "ite", (c, a, b):
                v_c = eval_bool(c)
                v_a = eval_rat(a)
                v_b = eval_rat(b)
                ret = z3.If(v_c, v_a, v_b)
            case "sb_right", (a,):
                v_a = eval_rat(a)
                ret = v_a / (1 + v_a)
        assert isinstance(ret, z3.ArithRef), (type(ret).__name__, ret)
        return ret

    return eval_bool, eval_int, eval_rat


if False:
    e: BoolTerm = (
        "and",
        (
            ("lit", (True,)),
            ("var", ("foob",)),
        ),
    )
    print(semantics_z3(lambda _: z3.Bool("a"), lambda _: None, lambda _: None)[0](e))
